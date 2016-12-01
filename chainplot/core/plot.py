import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as op
from adjustText import adjust_text
from scipy.stats.kde import gaussian_kde

import chainplot.core.style as plot_style
from chainplot.utils.dict_tools import replace_dict, split_kwargs, britishdict, combine_dict
from chainplot.utils.string_tools import prettify


# Define some helper functions (should probably go into class as static methods tbh or in separate module)
def categorical_lookup(series):
    s = series.copy()
    unique_vals = sorted(np.unique(s))
    return dict(zip(unique_vals, np.arange(1, len(unique_vals) + 1)))


def facet_dimensions(number_of_plots):
    sqrt_nplots = np.sqrt(number_of_plots)
    side_length = int(sqrt_nplots)
    nrows, ncols = (side_length, side_length)

    if number_of_plots < 4:
        nrows = 1
        ncols = number_of_plots
    elif sqrt_nplots > (side_length + 0.5):
        nrows += 1
        ncols += 1
    elif sqrt_nplots > side_length:
        ncols += 1

    return nrows, ncols


class Plot:
    def __init__(self, data, mapping=None, labels=None, style=plot_style.themes['default'], **kwargs):
        self.data = data.copy()
        self.plot_data = self.data.copy()

        self.mapping = mapping
        self.fig = plt.figure(**kwargs)
        self.number_of_plots = None
        self.axes = None

        # Set plot defaults
        self.mapping = {
            'x': 'x',
            'y': 'y',
            'by': None,
            'shadow': False  # this isn't really a mapping; it should go somewhere else!
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

        # Set style
        if type(style) == str:
            style = plot_style.themes[style.lower()]

        self.style = combine_dict(
            plot_style.THEME_SNOW,
            style.copy()
        )

        self.fig.set_facecolor(self.style['fig']['background']['color'])

        # store model fitting parameters
        self.fitted_parameters = []

    def subset_data(self, subset_function):
        self.plot_data = self.data.loc[subset_function]
        return self

    def alter_data(self, data):
        self.plot_data = data
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
                # subtitle = sorted(self.plot_data[self.mapping['by']].unique())
                subtitle = sorted([s for s in self.pull_data('by', self.data).unique()])
                subtitle = [prettify(sub) for sub in subtitle]
            else:
                subtitle = prettify(labels['subtitle'])

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
                # Set tick label size
                for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                    label.set_fontsize(style['axes']['label']['size'])
                    label.set_fontsize(style['axes']['label']['size'])

                # Set background colour
                ax.set_axis_bgcolor(style['axes']['background']['color'])

                # Need to check row and column
                nrows, ncols = facet_dimensions(self.number_of_plots)

                if (i % ncols) == 0:
                    ax.set_ylabel(labels['ylab'], **style['axes']['title'])
                if (i // ncols) == (nrows - 1):
                    ax.set_xlabel(labels['xlab'], **style['axes']['title'])

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
                # could put this in style dict
                ax.tick_params(
                    axis='x',  # changes apply to the x-axis (could do outside of loop?)
                    which='both',  # both major and minor ticks are affected
                    bottom='on',  # ticks along the bottom edge are on
                    top='off'  # ticks along the top edge are off
                )

                # could put this in style dict
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

    def map(self, **kwargs):
        # `alter_mapping`?
        # Split generating axes and setting the mapping into different methods
        # create new class for managing mappings?

        # update mapping with existing
        kwargs = combine_dict(self.mapping, kwargs)

        by = kwargs['by']
        shadow = kwargs['shadow']

        if by is None:
            self.number_of_plots = 1

            # Create dummy variable for faceting (there must be a netter way than this, right?)
            by = 'by'
            kwargs['by'] = 'by'
            self.plot_data['by'] = 1
        else:
            self.number_of_plots = len(self.plot_data[by].unique())

        nrows, ncols = facet_dimensions(self.number_of_plots)

        # check if there are pre-existing axes
        if self.axes is None:
            axes = [plt.subplot(nrows, ncols, i) for i in range(1, self.number_of_plots + 1)]
            self.axes = axes

        self.mapping['by'] = by
        self.mapping['shadow'] = shadow

        for k, v in kwargs.items():
            self.mapping[k] = v

        # Set default labels
        for lab in ('x', 'y'):
            if self.labels[lab + 'lab'] is None:
                self.labels[lab + 'lab'] = prettify(self.mapping[lab])

        return self.apply_style()

    def pull_data(self, attr, plot_data):
        data = plot_data.copy()
        if attr in self.mapping.keys():
            mapped_attr = self.mapping[attr]

            if type(mapped_attr) == str:
                attr_data = data[mapped_attr]

            elif callable(mapped_attr):
                attr_data = mapped_attr(data)

            elif type(mapped_attr) in (int, float):
                attr_data = mapped_attr * np.ones(data.shape[0])

            else:
                ValueError('Variables must be mapped to data with either string references or functions')
        else:
            attr_data = None  # pd.Series([None])

        return attr_data

    def filter_plot_data(self, plot_data, mapkey, value):
        if value is None:
            return plot_data
        else:
            return plot_data.loc[plot_data[self.mapping[mapkey]] == value]

    # Layering methods

    def layer_points(self, categorical=None, lookup=None, **kwargs):
        categories = self.get_facet_variables()

        # argument handling (should really be moved to a method or class or something)
        kwargs, shadow_kwargs = split_kwargs(kwargs, 'shadow_')

        kwargs = britishdict(kwargs)
        kwargs = combine_dict(
            self.style['layers']['points'],
            kwargs
        )

        shadow_kwargs = britishdict(shadow_kwargs)
        shadow_kwargs = combine_dict(
            self.style['shadow_defaults']['points'],
            shadow_kwargs
        )

        for i, ax in enumerate(self.axes):
            if i < len(categories):

                # could use `self.subset_data` method here instead, perhaps
                subcat = categories[i]
                plot_data = self.get_facet_data(subcat)

                xdata = self.pull_data('x', plot_data)
                ydata = self.pull_data('y', plot_data)

                if categorical is 'x':
                    lookup = categorical_lookup(plot_data[self.mapping['x']]) if lookup is None else lookup
                    ax.scatter(xdata.replace(lookup), ydata, **kwargs)

                    if self.mapping['shadow'] is True:
                        shadow_data = self.plot_data.loc[lambda df: df[self.mapping['by']] != subcat]
                        shadow_x = shadow_data[self.mapping['x']]
                        shadow_y = shadow_data[self.mapping['y']]

                        ax.scatter(shadow_x.replace(lookup), shadow_y, **shadow_kwargs)

                elif categorical is 'y':
                    lookup = categorical_lookup(plot_data[self.mapping['y']]) if lookup is None else lookup
                    ax.scatter(xdata, ydata.replace(lookup), **kwargs)

                    if self.mapping['shadow'] is True:
                        shadow_data = self.plot_data.loc[lambda df: df[self.mapping['by']] != subcat]
                        shadow_x = shadow_data[self.mapping['x']]
                        shadow_y = shadow_data[self.mapping['y']]

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

    def layer_histogram(self, **kwargs):
        categories = sorted(self.plot_data[self.mapping['by']].unique())

        kwargs, shadow_kwargs = split_kwargs(kwargs, 'shadow_')

        kwargs = britishdict(kwargs)
        kwargs = combine_dict(
            self.style['layers']['histogram'],
            kwargs
        )

        shadow_kwargs = britishdict(shadow_kwargs)
        shadow_kwargs = combine_dict(
            self.style['shadow_defaults']['points'],
            shadow_kwargs
        )

        for kw in [kwargs, shadow_kwargs]:
            if 'binwidth' in kw.keys():
                x_range = self.data_range(dimension='x')
                kw['bins'] = np.arange(x_range[0], x_range[1], kw['binwidth'])
                kw.pop('binwidth', 0)

        for i, ax in enumerate(self.axes):
            if i < len(categories):

                subcat = categories[i]
                plot_data = self.plot_data.loc[lambda df: df[self.mapping['by']] == subcat]

                xdata = self.pull_data('x', plot_data)
                xdata = [x for x in xdata if np.isfinite(x)]

                ax.hist(xdata, **kwargs)

                if self.mapping['shadow'] is True:
                    shadow_data = self.plot_data.loc[lambda df: df[self.mapping['by']] != subcat]
                    shadow_x = shadow_data[self.mapping['x']]

                    ax.hist(shadow_x, **shadow_kwargs)

            else:
                ax.axis('off')

        return self.apply_style()

    def layer_density(self, **kwargs):
        categories = sorted(self.plot_data[self.mapping['by']].unique())

        kwargs = britishdict(kwargs)

        for i, ax in enumerate(self.axes):
            if i < len(categories):
                subcat = categories[i]
                plot_data = self.plot_data.loc[lambda df: df[self.mapping['by']] == subcat]

                xdensity = gaussian_kde(self.pull_data('x', plot_data))

                x_range = self.data_range('x')
                xdata = np.linspace(x_range[0], x_range[1], 1000)
                ydata = xdensity.pdf(xdata)

                ax.fill(xdata, ydata, **kwargs)

        return self.apply_style()

    def layer_lines(self, **kwargs):
        categories = self.get_facet_variables()

        kwargs = britishdict(kwargs)
        kwargs = combine_dict(
            self.style['layers']['line'],
            kwargs
        )

        group_key = 'colour' if 'colour' in self.mapping.keys() else 'group'
        groups = np.unique(self.pull_data(group_key, self.data))

        colours = matplotlib.cm.get_cmap(self.style['colourmap'])

        for i, ax in enumerate(self.axes):
            if i < len(categories):
                subcat = categories[i]

                for gi, g in enumerate(groups):
                    plot_data = (
                        self.plot_data
                        .pipe(self.filter_plot_data, mapkey='by', value=subcat)
                        .pipe(self.filter_plot_data, mapkey=group_key, value=g)
                    )

                    xdata = self.pull_data('x', plot_data)
                    ydata = self.pull_data('y', plot_data)

                    if 'colour' in self.mapping.keys():
                        colour_index = 1 * gi / len(groups)  # scale to colourmap scale
                        kwargs = combine_dict(kwargs, {'color': colours(colour_index)})

                    kwargs = combine_dict(kwargs, {'label': g})  # add labels with overwriting (this could be an issue)
                    ax.plot(xdata, ydata, **kwargs)

            else:
                ax.axis('off')

        return self.apply_style()

    def layer_segments(self, **kwargs):
        categories = sorted(self.plot_data[self.mapping['by']].unique())

        kwargs = britishdict(kwargs)

        for i, ax in enumerate(self.axes):
            if i < len(categories):

                subcat = categories[i]
                plot_data = self.plot_data.loc[lambda df: df[self.mapping['by']] == subcat].reset_index(drop=True)

                xdata = self.pull_data('x', plot_data)
                ydata = self.pull_data('y', plot_data)
                x2data = self.pull_data('x2', plot_data)
                y2data = self.pull_data('y2', plot_data)

                for xi in range(len(xdata)):
                    x = (xdata[xi], x2data[xi])
                    y = (ydata[xi], y2data[xi])

                    ax.plot(x, y, **kwargs)

            else:
                ax.axis('off')

        return self.apply_style()

    def layer_text(self, categorical=None, lookup=None, check_overlap=False, **kwargs):
        categories = self.get_facet_variables()

        kwargs = britishdict(kwargs)
        kwargs, adjust_kwargs = split_kwargs(kwargs, 'adj_')

        for i, ax in enumerate(self.axes):
            if i < len(categories):

                subcat = categories[i]
                plot_data = self.get_facet_data(subcat).reset_index()

                xdata = self.pull_data('x', plot_data)
                ydata = self.pull_data('y', plot_data)
                txt_data = self.pull_data('label', plot_data)

                if categorical is 'x':
                    lookup = categorical_lookup(plot_data[self.mapping['x']]) if lookup is None else lookup
                    xdata = xdata.replace(lookup)
                if categorical is 'y':
                    lookup = categorical_lookup(plot_data[self.mapping['y']]) if lookup is None else lookup
                    ydata = ydata.replace(lookup)

                if check_overlap is True:
                    texts = []
                    for txt_i, txt in enumerate(txt_data):
                        texts.append(ax.text(xdata[txt_i], ydata[txt_i], str(txt), **kwargs))

                    adjust_text(texts, xdata, ydata, ax=ax, **adjust_kwargs)
                else:
                    for txt_i, txt in enumerate(txt_data):
                        # annotate is preferred else text isn't limited by axes limits
                        ax.annotate(str(txt), (xdata[txt_i], ydata[txt_i]), **kwargs)

            else:
                ax.axis('off')

        return self.apply_style()

    def layer_errorbars(self, categorical=None, **kwargs):
        categories = sorted(self.plot_data[self.mapping['by']].unique())

        kwargs = britishdict(kwargs)

        for i, ax in enumerate(self.axes):
            if i < len(categories):

                subcat = categories[i]
                plot_data = self.plot_data.loc[lambda df: df[self.mapping['by']] == subcat]

                xdata = self.pull_data('x', plot_data)
                ydata = self.pull_data('y', plot_data)
                xerror = self.pull_data('x_error', plot_data)
                yerror = self.pull_data('y_error', plot_data)

                if categorical is 'x':
                    lookup = categorical_lookup(plot_data[self.mapping['x']])
                    ax.errorbar(xdata.replace(lookup), ydata, xerr=xerror, yerr=yerror, **kwargs)

                elif categorical is 'y':
                    lookup = categorical_lookup(plot_data[self.mapping['y']])
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

    def layer_ribbons(self, categorical=None, **kwargs):
        categories = sorted(self.plot_data[self.mapping['by']].unique())

        kwargs = britishdict(kwargs)

        for i, ax in enumerate(self.axes):
            if i < len(categories):

                subcat = categories[i]
                plot_data = self.plot_data.loc[lambda df: df[self.mapping['by']] == subcat]

                xdata = self.pull_data('x', plot_data)
                y_lo_data = self.pull_data('y_lower', plot_data)  # could change these labels to be same as errorbars
                y_up_data = self.pull_data('y_upper', plot_data)  # ... or change error bars (probably preferable)

                lookup = None
                categorical = None
                ax.fill_between(xdata, y1=y_lo_data, y2=y_up_data, **kwargs)

            else:
                ax.axis('off')

        if categorical in ('x', 'y'):
            self.discrete_axis(categorical, lookup)

        return self.apply_style()

    def layer_heatmap(self, **kwargs):
        categories = self.get_facet_variables()

        kwargs, _ = split_kwargs(kwargs, 'shadow_')

        kwargs = britishdict(kwargs)
        kwargs = combine_dict(
            self.style['layers']['heatmap'],
            kwargs
        )
        kwargs = combine_dict(
            {'cmap': self.style['scales']['cmap']},
            kwargs
        )

        for i, ax in enumerate(self.axes):
            if i < len(categories):

                # could use `self.subset_data` method here instead, perhaps
                subcat = categories[i]
                plot_data = self.get_facet_data(subcat)

                # reshape data into matrix
                plot_matrix = pd.pivot_table(
                    plot_data,
                    index=self.mapping['y'],
                    columns=self.mapping['x'],
                    values=self.mapping['colour'],
                    fill_value=0  # could include this as a kwarg...
                )

                ax.imshow(plot_matrix, **kwargs)

            else:
                ax.axis('off')

        return self.apply_style()

    def layer_calcline(self, func, **kwargs):
        categories = sorted(self.plot_data[self.mapping['by']].unique())
        kwargs = britishdict(kwargs)

        for i, ax in enumerate(self.axes):
            if i < len(categories):
                xlim = ax.get_xlim()
                xline = np.linspace(xlim[0], xlim[1], 500)
                yline = [func(x) for x in xline]

                ax.plot(xline, yline, **kwargs)

                ax.set_xlim(xlim)

        return self

    def layer_refline(self, slope=None, intercept=None, invert=False, **kwargs):
        categories = sorted(self.plot_data[self.mapping['by']].unique())
        kwargs = britishdict(kwargs)
        kwargs = combine_dict(
            self.style['layers']['refline'],
            kwargs
        )

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

    def layer_vline(self, intercept=0, y_range=None, annotation='', **kwargs):
        categories = sorted(self.plot_data[self.mapping['by']].unique())
        kwargs = britishdict(kwargs)
        kwargs = combine_dict(
            self.style['layers']['refline'],
            kwargs
        )

        for i, ax in enumerate(self.axes):
            if i < len(categories):
                subcat = categories[i]
                plot_data = self.plot_data.loc[lambda df: df[self.mapping['by']] == subcat]

                if callable(intercept):
                    xintercept = intercept(plot_data[self.mapping['x']])
                else:
                    xintercept = intercept

                yline = ax.get_ylim() if y_range is None else y_range
                xline = [xintercept, xintercept]
                ax.plot(xline, yline, **kwargs)
                ax.annotate(annotation, (xline[1], yline[1]),
                            va='center', ha='right',
                            color=self.style['axes']['title']['color'])
                ax.set_ylim(yline)

        return self

    def layer_hline(self, intercept=0, x_range=None, annotation='', **kwargs):
        categories = sorted(self.plot_data[self.mapping['by']].unique())
        kwargs = britishdict(kwargs)
        kwargs = combine_dict(
            self.style['layers']['refline'],
            kwargs
        )

        for i, ax in enumerate(self.axes):
            if i < len(categories):
                subcat = categories[i]
                plot_data = self.plot_data.loc[lambda df: df[self.mapping['by']] == subcat]

                if callable(intercept):
                    yintercept = intercept(plot_data[self.mapping['y']])
                else:
                    yintercept = intercept

                xline = ax.get_xlim() if x_range is None else x_range
                yline = [yintercept, yintercept]

                ax.plot(xline, yline, **kwargs)
                ax.annotate(annotation, (xline[1], yline[1]),
                            va='bottom', ha='right',
                            color=self.style['axes']['title']['color'])
                ax.set_xlim(xline)

        return self

    def layer_trendline(self, objective_function=lambda x, a, b: a * x + b, **kwargs):

        categories = sorted(self.plot_data[self.mapping['by']].unique())
        kwargs = britishdict(kwargs)
        kwargs = combine_dict(
            self.style['layers']['trendline'],
            kwargs
        )

        for i, ax in enumerate(self.axes):
            subcat = categories[i]
            plot_data = self.plot_data.loc[lambda df: df[self.mapping['by']] == subcat]

            xdata = self.pull_data('x', plot_data)
            ydata = self.pull_data('y', plot_data)

            params, _ = op.curve_fit(objective_function, xdata, ydata)

            self.fitted_parameters.append(params)

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
        return min(self.plot_data[self.mapping[dimension]]), max(self.plot_data[self.mapping[dimension]])

    def check_mapping(self, mapname):
        try:
            self.mapping[mapname]
        except KeyError:
            print('No mapping for required argument \'' + mapname + '\' specified.')

    def get_facet_variables(self):
        facet_column = self.mapping['by']
        if facet_column is None:
            return None
        else:
            # should check for errors here
            return sorted(self.plot_data[facet_column].unique())

    def get_facet_data(self, subcategory):
        # redundant with `filter_plot_data`
        facet_column = self.mapping['by']
        if facet_column is None:
            return self.plot_data
        else:
            return self.plot_data.loc[lambda df: df[facet_column] == subcategory]

    # Styling

    def add_legend(self, legend_axes=(0,), *args, **kwargs):
        kwargs = combine_dict(
            self.style['legend'],
            kwargs
        )
        for i in legend_axes:
            ax = self.axes[i]
            ax.legend(*args, **kwargs)
        return self

    def tight_layout(self, *args, **kwargs):
        self.fig.tight_layout(*args, **kwargs)
        return self

    def set_xlim(self, lims, **kwargs):
        # `set_xlim(...)` ?
        for ax in self.axes:
            ax.set_xlim(lims, **kwargs)
        return self

    def set_ylim(self, lims, **kwargs):
        for ax in self.axes:
            ax.set_ylim(lims, **kwargs)
        return self

    def set_xlab(self, lab):
        self.labels['xlab'] = lab
        return self.apply_style()

    def set_ylab(self, lab):
        self.labels['ylab'] = lab
        return self.apply_style()

    def set_subtitle(self, subtitle=None, **kwargs):
        # should it be plural (`set_subtitles(...)`)?
        for k, v in self.style['subtitle'].items():
            if k not in kwargs.keys():
                kwargs[k] = v

        if subtitle is None:
            if len(self.axes) > 1:
                subtitle = sorted(self.plot_data[self.mapping['by']].unique())
                subtitle = [prettify(sub) for sub in subtitle]
            else:
                subtitle = ''

        if isinstance(subtitle, (list, tuple)):
            for sub, ax in zip(subtitle, self.axes):
                ax.set_title(sub, **kwargs)
        else:
            for ax in self.axes:
                ax.set_title(subtitle, **kwargs)

        return self

    def set_title(self, title=None, **kwargs):

        for k, v in self.style['title'].items():
            if k not in kwargs.keys():
                kwargs[k] = v

        self.fig.suptitle(title, **kwargs)

        return self

    def pipe(self, func, obj=None, *args, **kwargs):
        if obj is None:
            func(self, *args, **kwargs)

        elif obj == 'data':
            self.plot_data = func(self.plot_data, *args, **kwargs)

        elif obj == 'fig':
            func(self.fig, *args, **kwargs)

        elif obj == 'ax':
            for ax in self.axes:
                func(ax, *args, **kwargs)

        return self

    def save(self, file_location, **kwargs):
        self.fig.savefig(file_location, **kwargs)

    def show(self):
        self.fig.show()
