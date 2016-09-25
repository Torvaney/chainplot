import matplotlib.pyplot as plt
import numpy as np

# NOTES
# Need to think about how flexible vs just perosnal use?


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


def apply_style_continuous(ax):
    ax.spines["bottom"].set_color('dimgray')
    ax.spines["left"].set_color('dimgray')
    ax.tick_params(colors='dimgray')
    # Remove top and bottom spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Remove extra ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # Set background colour
    ax.set_axis_bgcolor('snow')


def apply_style_discrete(ax, lookup, categorical):
    if categorical == 'x':
        ax.set_xticks(list(lookup.values()))
        ax.set_xticklabels(list(lookup.keys()))
    elif categorical == 'y':
        ax.set_yticks(list(lookup.values()))
        ax.set_yticklabels(list(lookup.keys()))

    # Change axes colour
    ax.spines["bottom"].set_color('dimgray')
    ax.spines["left"].set_color('dimgray')
    ax.tick_params(colors='dimgray')

    # Remove top and bottom spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Remove extra ticks
    if categorical == 'x':
        ax.get_xaxis().tick_bottom()
        ax.tick_params(
            axis='x',  # changes apply to the x-axis (could do outside of loop?)
            which='both',  # both major and minor ticks are affected
            left='on',  # ticks along the bottom edge are off
            right='off'  # ticks along the top edge are off
        )

        ax.xaxis.grid(True, which='major', color='dimgray', linestyle='dotted')
        ax.set_xlim([min(list(lookup.values())) - 1, max(list(lookup.values())) + 1])
    elif categorical == 'y':
        ax.get_yaxis().tick_left()
        ax.tick_params(
            axis='y',  # changes apply to the y-axis
            which='both',  # both major and minor ticks are affected
            left='on',  # ticks along the bottom edge are off
            right='off'  # ticks along the top edge are off
        )

        ax.yaxis.grid(True, which='major', color='dimgray', linestyle='dotted')
        ax.set_ylim(
            (min(list(lookup.values())) - 1,
             max(list(lookup.values())) + 1)
        )

    # Set background colour
    ax.set_axis_bgcolor('snow')


def apply_style_blank(ax):
    ax.axis('off')


class Plot:
    def __init__(self, data, aes=None, **kwargs):
        self.data = data
        self.aes = aes
        self.fig = plt.figure(**kwargs)
        self.axes = None
        self.aes = {
            'x': 'x',
            'y': 'y',
            'by': 'by'
        }

        self.fig.set_facecolor('snow')

    def aesthetics(self, x=None, y=None, by=None, **kwargs):
        if by is None:
            number_of_plots = 1

            # Create dummy variable for faceting
            by = 'by'
            self.data['by'] = 1
        else:
            number_of_plots = len(self.data[by].unique())

        nrows, ncols = facet_dimensions(number_of_plots)

        axes = [plt.subplot(nrows, ncols, i) for i in range(1, number_of_plots + 1)]
        self.axes = axes

        self.aes['x'] = x
        self.aes['y'] = y
        self.aes['by'] = by

        for k, v in kwargs.items():
            self.aes[k] = v

        return self

    def points(self, categorical=None, aes=None, **kwargs):
        categories = sorted(self.data[self.aes['by']].unique())

        # Fix awful americanisms
        if 'colour' in kwargs.keys():
            kwargs['color'] = kwargs['colour']
            kwargs.pop('colour', 0)
        if 'edgecolour' in kwargs.keys():
            kwargs['edgecolor'] = kwargs['edgecolour']
            kwargs.pop('edgecolour', 0)

        for i, ax in enumerate(self.axes):
            if i < len(categories):

                subcat = categories[i]
                plot_data = self.data.loc[lambda df: df[self.aes['by']] == subcat]

                xdata = plot_data[self.aes['x']]
                ydata = plot_data[self.aes['y']]

                if categorical is None:
                    # could create a kwargs dict dynamically or through a loop or something
                    ax.scatter(xdata, ydata, **kwargs)
                    apply_style_continuous(ax)

                elif categorical is 'x':
                    xlookup = categorical_lookup(plot_data[self.aes['x']])
                    ax.scatter(xdata.replace(xlookup), ydata, **kwargs)
                    apply_style_discrete(ax, xlookup, categorical)

                elif categorical is 'y':
                    ylookup = categorical_lookup(plot_data[self.aes['y']])
                    ax.scatter(xdata, ydata.replace(ylookup), **kwargs)
                    apply_style_discrete(ax, ylookup, categorical)

            else:
                apply_style_blank(ax)

        return self

    def histogram(self, aes=None, **kwargs):
        categories = sorted(self.data[self.aes['by']].unique())

        for i, ax in enumerate(self.axes):
            if i < len(categories):

                subcat = categories[i]
                plot_data = self.data.loc[lambda df: df[self.aes['by']] == subcat]

                xdata = plot_data[self.aes['x']]

                ax.hist(xdata, **kwargs)
                apply_style_continuous(ax)

            else:
                apply_style_blank(ax)

        return self

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

    def hline(self, intercept=0, **kwargs):
        yline = [min(self.data[self.aes['y']]), max(self.data[self.aes['y']])]
        xline = [intercept, intercept]

        for ax in self.axes:
            ax.plot(xline, yline, **kwargs)

        return self

    def vline(self, intercept=0, **kwargs):
        xline = [min(self.data[self.aes['x']]), max(self.data[self.aes['x']])]
        yline = [intercept, intercept]

        for ax in self.axes:
            ax.plot(xline, yline, **kwargs)

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

    def title(self, title=None, **kwargs):
        self.fig.suptitle(title, **kwargs)