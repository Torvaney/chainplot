# NOTES
* Need to think about how flexible vs just personal use/convenience?
  For some level of flexibility it'll make more sense just to start from scratch w/ graphics of grammar
* How much am I just going to rip off ggplot? And how much to adapt to a more 'pythonic' approach?
* How do I set default styles/subtitles etc in a natural manner?
* new `layer_*` names actually kind of make it harder to read?
* Given the similarities, can I abstract the layering somehow? (maybe a layer class or something?)

# Stuff to add:
* Add default styling for layers to style.py
* Add ordering for facets/categorical variables
* Move non-plotting functions to another directory
* Could have style class rather than just all in a dict?
* Add callables in mapping (e.g. lambda functions). Could go well with separate mapping class idea
* Callables for labels and subtitles
* Add requirements.txt
* Add facet grid (maybe change method of faceting? idk)
* Should categorical lookups be set somewhere as attributes? e.g. `self.lookups = {'y': None}`
* Get categorical variables automatically
* rename vline and hline `yline` and `xline`?
* Set facets in a separate method - I can definitely take the axis creation parts out of `map` and put them in ...
  ... elsewhere and then call method if layer is none and no axes are inintialised.
* need to sort all styling (blech)
