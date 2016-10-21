

THEME_SNOW = {
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
    'categorical': {
        'grid': {
            'which': 'major',
            'color': 'dimgray',
            'linestyle': 'dotted'
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
        'line': {
            'color': 'black'
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
        'color': 'dimgray'
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
    'categorical': {
        'grid': {
            'which': 'major',
            'color': 'dimgray',
            'linestyle': 'dotted'
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
        'line': {
            'color': 'black'
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
        'color': 'dimgray'
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
        'frameon': True
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
        'line': {
            'color': 'black'
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
    'default': GGPLOT,
    'theme_bw': THEME_BW,
    'theme_snow': THEME_SNOW,
    'ggplot': GGPLOT
}
