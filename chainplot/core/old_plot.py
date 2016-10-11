import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
from adjustText import adjust_text
from scipy.stats.kde import gaussian_kde

import chainplot.core.style as plot_style
from chainplot.utils.dict_tools import replace_dict, split_kwargs, britishdict, combine_dict


# NOTES
# Need to think about how flexible vs just personal use/convenience?
#   For some level of flexibility it'll make more sense just to start from scratch w/ graphics of grammar
# How much am I just going to rip off ggplot? And how much to adapt to a more 'pythonic' approach?
# How do I set default styles/subtitles etc in a natural manner?

# Stuff to add:
# Add ordering for facets/categorical variables
# Change names of methods to `layer_*`
# Move non-plotting functions to another directory


# Define some helper functions (should probably go into class as static methods tbh)
def categorical_lookup(series):
    s = series.copy()
    unique_vals = sorted(s.unique())
    return dict(zip(unique_vals, np.arange(1, len(unique_vals) + 1)))


def facet_dimensions(number_of_plots):
    sqrt_nplots = np.sqrt(number_of_plots)
    side_length = int(sqrt_nplots)
    nrows, ncols = (side_length, side_length)

    if sqrt_nplots > (side_length + 0.5):
        nrows += 1
        ncols += 1
    elif sqrt_nplots > side_length:
        ncols += 1

    return nrows, ncols


class Plot:
    # Change methods to copy object before modifying + returning
    # Make code more EAFP than LBYL?
    def __init__(self, data, aes=None, labels=None, style=None, **kwargs):
        self.data = data.copy()
        self.aes = aes
        self.fig = plt.figure(**kwargs)
        self.number_of_plots = None
        self.axes = None

        # Set plot defaults
        self.aes = {
            'x': 'x',
            'y': 'y',
            'by': None,
            'shadow': False
        }

        if labels is None:
            self.labels = {
                'title': None,
                'subtitle': None,
                'xlab': None,
                'ylab': None
            }
        else:
            self.labels = labels.copy()

        if style is None:
            self.style = plot_style.THEME_SNOW.copy()
        else:
            self.style = style.copy()

        self.fig.set_facecolor(self.style['fig']['background']['color'])

    def alter_data(self, data):
        self.data = data
        return self

    def alter_style(self, newstyle):
        self.style = combine_dict(self.style, newstyle)
        return self.apply_style()

    def apply_style(self, style=None):
        if style is None:
            style = self.style

        labels = replace_dict(self.labels, val=None, replacement=' ')

        if self.axes is not None:
            if len(self.axes) > 1 and self.labels['subtitle'] is None:
                subtitle = sorted(self.data[self.aes['by']].unique())
            else:
                subtitle = labels['subtitle']

            for i, ax in enumerate(self.axes):
                ax.spines['bottom'].set_color(style['axes']['spines']['color'])
                ax.spines['left'].set_color(style['axes']['spines']['color'])
                ax.tick_params(colors=style['axes']['spines']['color'])

                # Remove top and bottom spines
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                # Remove extra ticks
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()

                # Set background colour
                ax.set_axis_bgcolor(style['axes']['background']['color'])

                # Need to check row and column
                nrows, ncols = facet_dimensions(self.number_of_plots)

                if (i % ncols) == 0:
                    ax.set_ylabel(labels['ylab'], **style['axes']['text'])
                if (i // ncols) == (nrows - 1):
                    ax.set_xlabel(labels['xlab'], **style['axes']['text'])

                ax.set_title(subtitle[i], **style['subtitle'])

        self.fig.set_facecolor(self.style['fig']['background']['color'])

        return self

    def discrete_axis(self, categorical, lookup):
        self.labels[categorical + 'lab'] = ''

        for ax in self.axes:
            # (I can definitely make all this much more dynamic by leveraging `categorical` variable)
            if categorical == 'x':
                ax.set_xticks(list(lookup.values()))
                ax.set_xticklabels(list(lookup.keys()))

            elif categorical == 'y':
                ax.set_yticks(list(lookup.values()))
                ax.set_yticklabels(list(lookup.keys()))

            # Remove extra ticks
            if categorical == 'x':

                ax.spines["bottom"].set_visible(False)

                ax.get_yaxis().tick_left()
                ax.tick_params(
                    axis='x',  # changes apply to the x-axis (could do outside of loop?)
                    which='both',  # both major and minor ticks are affected
                    bottom='on',  # ticks along the bottom edge are on
                    top='off'  # ticks along the top edge are off
                )

                ax.xaxis.grid(True, which='major', color='dimgray', linestyle='dotted')
                ax.set_xlim(
                    [min(list(lookup.values())) - 1,
                     max(list(lookup.values())) + 1]
                )

            elif categorical == 'y':

                ax.spines["left"].set_visible(False)

                ax.get_xaxis().tick_bottom()
                ax.tick_params(
                    axis='y',  # changes apply to the y-axis
                    which='both',  # both major and minor ticks are affected
                    left='on',  # ticks along the left edge are on
                    right='off'  # ticks along the right edge are off
                )

                ax.yaxis.grid(True, which='major', color='dimgray', linestyle='dotted')
                ax.set_ylim(
                    (min(list(lookup.values())) - 1,
                     max(list(lookup.values())) + 1)
                )

        return self

    def aesthetics(self, **kwargs):
        # rename to `map`
        # Make a new `mappings` class to make this code cleaner

        # update aesthetics
        kwargs = combine_dict(self.aes, kwargs)

        by = kwargs['by']
        shadow = kwargs['shadow']

        if by is None:
            self.number_of_plots = 1

            # Create dummy variable for faceting (there must be a netter way than this, right?)
            by = 'by'
            kwargs['by'] = 'by'
            self.data['by'] = 1
        else:
            self.number_of_plots = len(self.data[by].unique())

        nrows, ncols = facet_dimensions(self.number_of_plots)

        # check if there are pre-existing axes
        if self.axes is None:
            axes = [plt.subplot(nrows, ncols, i) for i in range(1, self.number_of_plots + 1)]
            self.axes = axes

        self.aes['by'] = by
        self.aes['shadow'] = shadow

        for k, v in kwargs.items():
            self.aes[k] = v

        # Set default labels
        for lab in ('x', 'y'):
            if self.labels[lab + 'lab'] is None:
                self.labels[lab + 'lab'] = self.aes[lab]

        return self.apply_style()

    def points(self, categorical=None, lookup=None, **kwargs):
        categories = sorted(self.data[self.aes['by']].unique())

        kwargs, shadow_kwargs = split_kwargs(kwargs, 'shadow_')

        kwargs = britishdict(kwargs)
        shadow_kwargs = britishdict(shadow_kwargs)

        shadow_kwargs = combine_dict(
            self.style['shadow_defaults']['points'],
            shadow_kwargs
        )

        for i, ax in enumerate(self.axes):
            if i < len(categories):

                subcat = categories[i]
                plot_data = self.data.loc[lambda df: df[self.aes['by']] == subcat]

                xdata = plot_data[self.aes['x']]
                ydata = plot_data[self.aes['y']]

                if categorical is 'x':
                    lookup = categorical_lookup(plot_data[self.aes['x']]) if lookup is None else lookup
                    ax.scatter(xdata.replace(lookup), ydata, **kwargs)

                    if self.aes['shadow'] is True:
                        shadow_data = self.data.loc[lambda df: df[self.aes['by']] != subcat]
                        shadow_x = shadow_data[self.aes['x']]
                        shadow_y = shadow_data[self.aes['y']]

                        ax.scatter(shadow_x.replace(lookup), shadow_y, **shadow_kwargs)

                elif categorical is 'y':
                    lookup = categorical_lookup(plot_data[self.aes['y']]) if lookup is None else lookup
                    ax.scatter(xdata, ydata.replace(lookup), **kwargs)

                    if self.aes['shadow'] is True:
                        shadow_data = self.data.loc[lambda df: df[self.aes['by']] != subcat]
                        shadow_x = shadow_data[self.aes['x']]
                        shadow_y = shadow_data[self.aes['y']]

                        ax.scatter(shadow_x, shadow_y.replace(lookup), **shadow_kwargs)

                else:  # if both variables are continuous
                    lookup = None
                    categorical = None
                    ax.scatter(xdata, ydata, **kwargs)

            else:
                ax.axis('off')

        if categorical in ('x', 'y'):
            self.discrete_axis(categorical, lookup)

        return self.apply_style()

    def histogram(self, **kwargs):
        categories = sorted(self.data[self.aes['by']].unique())

        kwargs, shadow_kwargs = split_kwargs(kwargs, 'shadow_')

        kwargs = britishdict(kwargs)
        shadow_kwargs = britishdict(shadow_kwargs)

        shadow_kwargs = combine_dict(
            self.style['shadow_defaults']['points'],
            shadow_kwargs
        )

        for kw in [kwargs, shadow_kwargs]:
            if 'binwidth' in kw.keys():
                xrange = self.data_range(dimension='x')
                kw['bins'] = np.arange(xrange[0], xrange[1], kw['binwidth'])
                kw.pop('binwidth', 0)

        for i, ax in enumerate(self.axes):
            if i < len(categories):

                subcat = categories[i]
                plot_data = self.data.loc[lambda df: df[self.aes['by']] == subcat]

                xdata = plot_data[self.aes['x']]

                ax.hist(xdata, **kwargs)

                if self.aes['shadow'] is True:
                    shadow_data = self.data.loc[lambda df: df[self.aes['by']] != subcat]
                    shadow_x = shadow_data[self.aes['x']]

                    ax.hist(shadow_x, **shadow_kwargs)

            else:
                ax.axis('off')

        return self.apply_style()

    def density(self, **kwargs):
        categories = sorted(self.data[self.aes['by']].unique())

        kwargs = britishdict(kwargs)

        for i, ax in enumerate(self.axes):
            if i < len(categories):
                subcat = categories[i]
                plot_data = self.data.loc[lambda df: df[self.aes['by']] == subcat]

                xdensity = gaussian_kde(plot_data[self.aes['x']])

                xrange = self.data_range('x')
                xdata = np.linspace(xrange[0], xrange[1], 1000)
                ydata = xdensity.pdf(xdata)

                ax.plot(xdata, ydata, **kwargs)

        return self.apply_style()

    def lines(self, **kwargs):
        categories = sorted(self.data[self.aes['by']].unique())

        if 'colour' in kwargs.keys():
            kwargs['color'] = kwargs['colour']
            kwargs.pop('colour', 0)

        for i, ax in enumerate(self.axes):
            if i < len(categories):

                subcat = categories[i]
                plot_data = self.data.loc[lambda df: df[self.aes['by']] == subcat]

                xdata = plot_data[self.aes['x']]
                ydata = plot_data[self.aes['y']]

                # could create a kwargs dict dynamically or through a loop or something
                ax.plot(xdata, ydata, **kwargs)

            else:
                ax.axis('off')

        return self.apply_style()

    def segments(self, **kwargs):
        categories = sorted(self.data[self.aes['by']].unique())

        if 'colour' in kwargs.keys():
            kwargs['color'] = kwargs['colour']
            kwargs.pop('colour', 0)

        for i, ax in enumerate(self.axes):
            if i < len(categories):

                subcat = categories[i]
                plot_data = self.data.loc[lambda df: df[self.aes['by']] == subcat].reset_index(drop=True)

                xdata = plot_data[self.aes['x']]
                ydata = plot_data[self.aes['y']]
                x2data = plot_data[self.aes['x2']]
                y2data = plot_data[self.aes['y2']]

                for xi in range(len(xdata)):
                    x = (xdata[xi], x2data[xi])
                    y = (ydata[xi], y2data[xi])

                    ax.plot(x, y, **kwargs)

            else:
                ax.axis('off')

        return self.apply_style()

    def label(self, categorical=None, lookup=None, check_overlap=False, **kwargs):
        # rename to `text`
        categories = sorted(self.data[self.aes['by']].unique())

        kwargs = britishdict(kwargs)
        kwargs, adjust_kwargs = split_kwargs(kwargs, 'adj_')

        for i, ax in enumerate(self.axes):
            if i < len(categories):

                subcat = categories[i]
                plot_data = self.data.loc[lambda df: df[self.aes['by']] == subcat].reset_index()

                xdata = plot_data[self.aes['x']]
                ydata = plot_data[self.aes['y']]
                txt_data = plot_data[self.aes['label']]

                if categorical is 'x':
                    lookup = categorical_lookup(plot_data[self.aes['x']]) if lookup is None else lookup
                    xdata = xdata.replace(lookup)
                if categorical is 'y':
                    lookup = categorical_lookup(plot_data[self.aes['y']]) if lookup is None else lookup
                    ydata = ydata.replace(lookup)

                texts = []
                for txt_i, txt in enumerate(txt_data):
                    # ax.annotate(str(txt), (xdata[txt_i], ydata[txt_i]), **kwargs)  # old method
                    texts.append(ax.text(xdata[txt_i], ydata[txt_i], str(txt), **kwargs))

                if check_overlap:
                    adjust_text(texts, xdata, ydata, ax=ax, **adjust_kwargs)

            else:
                ax.axis('off')

        return self.apply_style()

    def error_bars(self, categorical=None, **kwargs):
        categories = sorted(self.data[self.aes['by']].unique())

        kwargs = britishdict(kwargs)

        for i, ax in enumerate(self.axes):
            if i < len(categories):

                subcat = categories[i]
                plot_data = self.data.loc[lambda df: df[self.aes['by']] == subcat]

                xdata = plot_data[self.aes['x']]
                ydata = plot_data[self.aes['y']]
                xerror = plot_data[self.aes['x_error']] if ('x_error' in self.aes.keys()) else None
                yerror = plot_data[self.aes['y_error']] if ('y_error' in self.aes.keys()) else None

                if categorical is 'x':
                    lookup = categorical_lookup(plot_data[self.aes['x']])
                    ax.errorbar(xdata.replace(lookup), ydata, xerr=xerror, yerr=yerror, **kwargs)

                elif categorical is 'y':
                    lookup = categorical_lookup(plot_data[self.aes['y']])
                    ax.errorbar(xdata, ydata.replace(lookup), xerr=xerror, yerr=yerror, **kwargs)

                else:  # if both variables are continuous
                    lookup = None
                    categorical = None
                    ax.errorbar(xdata, ydata, xerr=xerror, yerr=yerror, **kwargs)

            else:
                ax.axis('off')

        if categorical in ('x', 'y'):
            self.discrete_axis(categorical, lookup)

        return self.apply_style()

    def calc_line(self, func, **kwargs):
        categories = sorted(self.data[self.aes['by']].unique())
        kwargs = britishdict(kwargs)

        for i, ax in enumerate(self.axes):
            if i < len(categories):
                xlim = ax.get_xlim()
                xline = np.linspace(xlim[0], xlim[1], 500)
                yline = [func(x) for x in xline]

                ax.plot(xline, yline, **kwargs)

                ax.set_xlim(xlim)

        return self

    def ref_line(self, slope=None, intercept=None, invert=False, **kwargs):
        categories = sorted(self.data[self.aes['by']].unique())
        kwargs = britishdict(kwargs)

        for i, ax in enumerate(self.axes):
            if i < len(categories):
                if not invert:
                    xline = ax.get_xlim()
                    yline = [i * slope + intercept for i in xline]
                else:
                    yline = ax.get_ylim()
                    xline = [i * slope + intercept for i in yline]

                ax.plot(xline, yline, **kwargs)

        return self

    def vline(self, intercept=0, y_range=None, annotation='', **kwargs):
        categories = sorted(self.data[self.aes['by']].unique())
        kwargs = britishdict(kwargs)

        for i, ax in enumerate(self.axes):
            if i < len(categories):
                subcat = categories[i]
                plot_data = self.data.loc[lambda df: df[self.aes['by']] == subcat]

                if callable(intercept):
                    xintercept = intercept(plot_data[self.aes['x']])
                else:
                    xintercept = intercept

                yline = ax.get_ylim() if y_range is None else y_range
                xline = [xintercept, xintercept]
                ax.plot(xline, yline, **kwargs)
                ax.annotate(annotation, (xline[1], yline[1]),
                            va='center', ha='right',
                            color=self.style['axes']['text']['color'])
                ax.set_ylim(yline)

        return self

    def hline(self, intercept=0, x_range=None, annotation='', **kwargs):
        categories = sorted(self.data[self.aes['by']].unique())
        kwargs = britishdict(kwargs)

        for i, ax in enumerate(self.axes):
            if i < len(categories):
                subcat = categories[i]
                plot_data = self.data.loc[lambda df: df[self.aes['by']] == subcat]

                if callable(intercept):
                    yintercept = intercept(plot_data[self.aes['y']])
                else:
                    yintercept = intercept

                xline = ax.get_xlim() if x_range is None else x_range
                yline = [yintercept, yintercept]

                ax.plot(xline, yline, **kwargs)
                ax.annotate(annotation, (xline[1], yline[1]),
                            va='bottom', ha='right',
                            color=self.style['axes']['text']['color'])
                ax.set_xlim(xline)

        return self

    def trendline(self, **kwargs):
        # edit to allow any polynomial
        categories = sorted(self.data[self.aes['by']].unique())
        kwargs = britishdict(kwargs)

        for i, ax in enumerate(self.axes):
            subcat = categories[i]
            plot_data = self.data.loc[lambda df: df[self.aes['by']] == subcat]

            xdata = plot_data[self.aes['x']]
            ydata = plot_data[self.aes['y']]
            m, c = np.polyfit(xdata, ydata, 1)

            xline = ax.get_xlim()
            yline = [m * x + c for x in xline]

            ax.plot(xline, yline, **kwargs)
            ax.set_xlim(xline)

        return self

    def fit_line(self, objective_function, **kwargs):
        categories = sorted(self.data[self.aes['by']].unique())
        kwargs = britishdict(kwargs)

        for i, ax in enumerate(self.axes):
            subcat = categories[i]
            plot_data = self.data.loc[lambda df: df[self.aes['by']] == subcat]

            xdata = plot_data[self.aes['x']]
            ydata = plot_data[self.aes['y']]

            params, _ = op.curve_fit(objective_function, xdata, ydata)

            xlims = ax.get_xlim()
            ylims = ax.get_ylim()

            x = np.linspace(xlims[0], xlims[1], 100)
            y = objective_function(x, *params)

            ax.plot(x, y, **kwargs)

            ax.set_xlim(xlims)
            ax.set_ylim(ylims)

        return self.apply_style()

    # Smaller helper functions

    def data_range(self, dimension):
        # could put this elsewhere in a different utils file?
        return min(self.data[self.aes[dimension]]), max(self.data[self.aes[dimension]])

    def check_mapping(self, mapname):
        try:
            self.aes[mapname]
        except KeyError:
            print('No mapping for required argument \'' + mapname + '\' specified.')

    # Styling

    def legend(self, legend_plots=(0,), *args, **kwargs):
        for i in legend_plots:
            ax = self.axes[i]
            ax.legend(*args, **kwargs)
        return self

    def tight_layout(self, *args, **kwargs):
        self.fig.tight_layout(*args, **kwargs)
        return self

    def xlim(self, lims, **kwargs):
        for ax in self.axes:
            ax.set_xlim(lims, **kwargs)
        return self

    def ylim(self, lims, **kwargs):
        for ax in self.axes:
            ax.set_ylim(lims, **kwargs)
        return self

    def xlab(self, lab, **kwargs):
        for k, v in self.style['axes']['text'].items():
            if k not in kwargs.keys():
                kwargs[k] = v

        self.labels['xlab'] = lab

        return self.apply_style()

    def ylab(self, lab, **kwargs):
        for k, v in self.style['axes']['text'].items():
            if k not in kwargs.keys():
                kwargs[k] = v

        self.labels['ylab'] = lab

        return self.apply_style()

    def subtitle(self, subtitle=None, **kwargs):
        for k, v in self.style['subtitle'].items():
            if k not in kwargs.keys():
                kwargs[k] = v

        if subtitle is None:
            if len(self.axes) > 1:
                subtitle = sorted(self.data[self.aes['by']].unique())
            else:
                subtitle = ''

        if type(subtitle) in (list, tuple):
            for i, ax in enumerate(self.axes):
                ax.set_title(subtitle[i], **kwargs)
        else:
            for ax in self.axes:
                ax.set_title(subtitle, **kwargs)

        return self

    def title(self, title=None, **kwargs):

        for k, v in self.style['title'].items():
            if k not in kwargs.keys():
                kwargs[k] = v

        self.fig.suptitle(title, **kwargs)

        return self

    def pipe(self, func, obj=None, *args, **kwargs):
        if obj is None:
            func(self, *args, **kwargs)

        elif obj == 'data':
            self.data = func(self.data, *args, **kwargs)

        elif obj == 'fig':
            func(self.fig, *args, **kwargs)

        elif obj == 'ax':
            for ax in self.axes:
                func(ax, *args, **kwargs)

        return self

    def save(self, file_location, **kwargs):
        self.fig.savefig(file_location, **kwargs)
