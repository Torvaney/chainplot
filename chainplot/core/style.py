

DEFAULT_STYLE = {
    'title': {
        'ha': 'left',
        'x': 0.125,
        'fontsize': 18,
        'color': 'black'
    },
    'subtitle': {
        'loc': 'left',
        'fontsize': 16,
        'color': 'dimgray'
    },
    'axes': {
        'text': {
            'fontsize': 16,
            'color': 'dimgray'
        },
        'spines': {
            'color': 'dimgray'
        },
        'background': {
            'color': 'snow'
        }
    },
    'legend': {
        'loc': 'upper left',
        'frameon': False
    },
    'scales': {
        'cmap': 'viridis'
    },
    'fig': {
        'background': {
            'color': 'snow'
        }
    },
    'shadow_defaults': {
        'points': {
            'color': 'k',
            'alpha': 0.2
        }
    },
    'layers': {
        'points': {
            'edgecolor': 'None',
            'color': 'black',
            'alpha': 1
        },
        'histogram': {
            'edgecolor': 'None',
            'color': 'black',
            'alpha': 1,
            'bins': 50
        },
        'refline': {
            'color': 'black',
            'ls': 'dotted'
        },
        'trendline': {
            'color': 'skyblue'
        }
    }
}

THEME_BW = {
    'title': {
        'ha': 'left',
        'x': 0.125,
        'fontsize': 18,
        'color': 'black'
    },
    'subtitle': {
        'loc': 'left',
        'fontsize': 16,
        'color': 'black'
    },
    'axes': {
        'text': {
            'fontsize': 16,
            'color': 'black'
        },
        'spines': {
            'color': 'black'
        },
        'background': {
            'color': 'white'
        }
    },
    'legend': {
        'loc': 'upper left',
        'frameon': False
    },
    'scales': {
        'cmap': 'viridis'
    },
    'fig': {
        'background': {
            'color': 'white'
        }
    },
    'shadow_defaults': {
        'points': {
            'color': 'k',
            'alpha': 0.2
        }
    },
    'layers': {
        'points': {
            'edgecolor': 'None',
            'color': 'black',
            'alpha': 1
        },
        'histogram': {
            'edgecolor': 'None',
            'color': 'black',
            'alpha': 1,
            'bins': 50
        },
        'refline': {
            'color': 'black',
            'ls': 'dotted'
        },
        'trendline': {
            'color': 'skyblue'
        }
    }
}

GGPLOT = {
    'title': {
        'ha': 'left',
        'x': 0.125,
        'fontsize': 18,
        'color': 'black'
    },
    'subtitle': {
        'loc': 'left',
        'fontsize': 16,
        'color': 'black'
    },
    'axes': {
        'text': {
            'fontsize': 16,
            'color': 'black'
        },
        'spines': {
            'color': 'black'
        },
        'background': {
            'color': 'whitesmoke'
        }
    },
    'legend': {
        'loc': 'upper left',
        'frameon': False
    },
    'scales': {
        'cmap': 'viridis'
    },
    'fig': {
        'background': {
            'color': 'white'
        }
    },
    'shadow_defaults': {
        'points': {
            'color': 'k',
            'alpha': 0.2
        }
    },
    'layers': {
        'points': {
            'edgecolor': 'None',
            'color': 'black',
            'alpha': 1
        },
        'histogram': {
            'edgecolor': 'None',
            'color': 'black',
            'alpha': 1,
            'bins': 50
        },
        'refline': {
            'color': 'black',
            'ls': 'dotted'
        },
        'trendline': {
            'color': 'skyblue'
        }
    }
}

themes = {
    'default': DEFAULT_STYLE,
    'theme_bw': THEME_BW,
    'ggplot': GGPLOT
}
