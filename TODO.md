# NOTES
* Given the similarities, can I abstract the layering somehow?
    * `proto_layer('points', ...)`, perhaps?
    * (maybe a layer class or something?)

# Stuff to add:
* Add default styling for layers to style.py
* Add ordering for facets/categorical variables
* Move non-plotting functions to another directory
* Could have style class rather than just all in a dict?
* Add callables in mapping (e.g. lambda functions).
    * Could go well with separate mapping class idea
* Callables for labels and subtitles
* Add requirements.txt
* Add facet grid (maybe change method of faceting? idk)
* Should categorical lookups be set somewhere as attributes? 
    * e.g. `self.lookups = {'y': None}`
* Get categorical variables automatically
* rename vline and hline `yline` and `xline`?
* Set facets in a separate method
    * I can definitely take the axis creation parts out of `map` 
    * call method if layer is none and no axes are inintialised.
* need to sort all styling (blech)
* do something about shadowing variables 
    * (shouldn't go in mapping; maybe as a `layer_*` argument instead?) 
* could make a separate `PlotStyle` class?
* Is the original faceting method (dummy var) closer to the true method?
    * Any single plot is just a plot in which data faceted the same
* could do `PlotMapping` class that `Plot` inherits from?
* add `scales` attribute(/class?) to help handle categorical variables
* Add scales mapping (colour, size)
* Add some decent documentation
* rename some methods
