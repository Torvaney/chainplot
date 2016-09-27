import matplotlib.pyplot as plt
import numpy as np
import chainplot.core.style as plot_style
from collections import Counter

# NOTES
# Need to think about how flexible vs just personal use/convenience?
#   For some level of flexibility it'll make more sense just to start from scratch w/ graphics of grammar
# How much am I just going to rip off ggplot? And how much to adapt to a more 'pythonic' approach?
# How do I set default styles/subtitles etc in a natural manner?


# Define some helper functions (should probably go into class as static methods tbh)
def categorical_lookup(series):
    unique_vals = sorted(series.unique())
    return dict(zip(unique_vals, np.arange(1, len(unique_vals) + 1)))


def facet_dimensions(number_of_plots):
    sqrt_nplots = np.sqrt(number_of_plots)
    side_length = int(sqrt_nplots)
    nrows, ncols = (side_length, side_length)

    if sqrt_nplots > (side_length + 0.5):
        nrows += 1
        ncols += 1
    elif sqrt_nplots > (side_length):
        nrows += 1

    return nrows, ncols


def check_dict(input_dict, val=None, replacement=''):
    output_dict = input_dict.copy()
    for k, v in output_dict.items():
        if v is val:
            output_dict[k] = replacement
    return output_dict


def to_kwargs(**kwargs):
    return kwargs


def britishdict(argdict):
    outdict = argdict.copy()

    # Fix awful americanisms
    if 'colour' in outdict.keys():
        outdict['color'] = outdict['colour']
        outdict.pop('colour', 0)
    if 'edgecolour' in outdict.keys():
        outdict['edgecolor'] = outdict['edgecolour']
        outdict.pop('edgecolour', 0)

    return outdict


def combine_dict(a, b):
    c = a.copy()
    try:
        for key, val in b.items():
            if type(val) == dict:
                c[key] = combine_dict(a[key], b[key])
            else:
                c[key] = val
    except AttributeError:  # In case oth isn't a dict
        return NotImplemented

    return c


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
            'by': 'by'
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
            self.style = plot_style.DEFAULT_STYLE.copy()
        else:
            self.style = style.copy()

        self.fig.set_facecolor(self.style['fig']['background']['color'])

    def alter_style(self, newstyle):
        self.style = combine_dict(self.style, newstyle)
        return self.apply_style()

    def apply_style(self, style=None):
        if style is None:
            style = self.style

        labels = check_dict(self.labels, val=None, replacement=' ')

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

    def aesthetics(self, x=None, y=None, by=None, **kwargs):
        if by is None:
            self.number_of_plots = 1

            # Create dummy variable for faceting
            by = 'by'
            self.data['by'] = 1
        else:
            self.number_of_plots = len(self.data[by].unique())

        nrows, ncols = facet_dimensions(self.number_of_plots)

        axes = [plt.subplot(nrows, ncols, i) for i in range(1, self.number_of_plots + 1)]
        self.axes = axes

        self.aes['x'] = x
        self.aes['y'] = y
        self.aes['by'] = by

        for k, v in kwargs.items():
            self.aes[k] = v

        # Set default labels
        for lab in ('x', 'y'):
            if self.labels[lab + 'lab'] is None:
                self.labels[lab + 'lab'] = self.aes[lab]

        return self.apply_style()

    def points(self, categorical=None, **kwargs):
        categories = sorted(self.data[self.aes['by']].unique())

        kwargs = britishdict(kwargs)

        # Add colour
        if 'colour' in self.aes.keys():
            col_cat = sorted(self.data[self.aes['colour']].unique())
            col_lookup = categorical_lookup(self.data[self.aes['colour']])
            kwargs['c'] = self.data[self.aes['colour']].replace(col_lookup)

        for i, ax in enumerate(self.axes):
            if i < len(categories):

                subcat = categories[i]
                plot_data = self.data.loc[lambda df: df[self.aes['by']] == subcat]

                xdata = plot_data[self.aes['x']]
                ydata = plot_data[self.aes['y']]

                if categorical is 'x':
                    lookup = categorical_lookup(plot_data[self.aes['x']])
                    ax.scatter(xdata.replace(lookup), ydata,
                               cmap=self.style['scales']['cmap'],
                               **kwargs)

                elif categorical is 'y':
                    lookup = categorical_lookup(plot_data[self.aes['y']])
                    ax.scatter(xdata, ydata.replace(lookup),
                               cmap=self.style['scales']['cmap'],
                               **kwargs)

                else:  # if both variables are continuous
                    lookup = None
                    categorical = None
                    ax.scatter(xdata, ydata,
                               cmap=self.style['scales']['cmap'],
                               **kwargs)

            else:
                ax.axis('off')

        if categorical in ('x', 'y'):
            self.discrete_axis(categorical, lookup)

        return self.apply_style()

    def histogram(self, **kwargs):
        categories = sorted(self.data[self.aes['by']].unique())

        kwargs = britishdict(kwargs)

        if 'binwidth' in kwargs.keys():
            xrange = self.data_range(dimension='x')
            kwargs['bins'] = np.arange(xrange[0], xrange[1], kwargs['binwidth'])
            kwargs.pop('binwidth', 0)

        for i, ax in enumerate(self.axes):
            if i < len(categories):

                subcat = categories[i]
                plot_data = self.data.loc[lambda df: df[self.aes['by']] == subcat]

                xdata = plot_data[self.aes['x']]

                ax.hist(xdata, **kwargs)

            else:
                ax.axis('off')

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

    def ref_line(self, slope=None, intercept=None, invert=False, **kwargs):
        if not invert:
            xline = [min(self.data[self.aes['x']]), max(self.data[self.aes['x']])]
            yline = [i * slope + intercept for i in xline]
        else:
            yline = [min(self.data[self.aes['y']]), max(self.data[self.aes['y']])]
            xline = [i * slope + intercept for i in yline]

        for ax in self.axes:
            ax.plot(xline, yline, **kwargs)

        return self

    def vline(self, intercept=0, yrange=None, annotation='', **kwargs):
        kwargs = britishdict(kwargs)

        if callable(intercept):  # change this to function for each subplot?
            intercept = intercept(self.data[self.aes['x']])

        for ax in self.axes:
            yline = ax.get_ylim() if yrange is None else yrange
            xline = [intercept, intercept]
            ax.plot(xline, yline, **kwargs)
            ax.annotate(annotation, (xline[1], yline[1]),
                        va='center', ha='right',
                        color=self.style['axes']['text']['color'])
            ax.set_ylim(yline)

        return self

    def hline(self, intercept=0, xrange=None, annotation='', **kwargs):
        kwargs = britishdict(kwargs)

        if callable(intercept):
            intercept = intercept(self.data[self.aes['y']])

        for ax in self.axes:
            xline = ax.get_xlim() if xrange is None else xrange
            yline = [intercept, intercept]
            ax.plot(xline, yline, **kwargs)
            ax.annotate(annotation, (xline[1], yline[1]),
                        va='bottom', ha='right',
                        color=self.style['axes']['text']['color'])
            ax.set_xlim(xline)

        return self

    def trendline(self, **kwargs):
        # edit to allow any polynomial
        categories = sorted(self.data[self.aes['by']].unique())

        for i, ax in enumerate(self.axes):
            subcat = categories[i]
            plot_data = self.data.loc[lambda df: df[self.aes['by']] == subcat]

            xdata = plot_data[self.aes['x']]
            ydata = plot_data[self.aes['y']]
            m, c = np.polyfit(xdata, ydata, 1)

            xline = [min(plot_data[self.aes['x']]), max(plot_data[self.aes['x']])]
            yline = [m * x + c for x in xline]

            ax.plot(xline, yline, **kwargs)

        return self

    # Smaller helper functions

    def data_range(self, dimension):
        return min(self.data[self.aes[dimension]]), max(self.data[self.aes[dimension]])

    # Styling

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

    def pipe(self, obj, func):
        if obj == 'fig':
            func(self.fig)

        if obj == 'ax':
            for ax in self.axes:
                func(ax)

        return self

    def save(self, file_location, **kwargs):
        self.fig.savefig(file_location, **kwargs)
