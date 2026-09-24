import { interpolateRgbBasis } from "d3";

// Rampe SÉQUENTIELLE à une teinte pour a_i (neutre -> critique), pas
// rouge->vert: un score est une magnitude, et rouge/vert est illisible en
// deutéranopie. Même rampe que src/dashboard/causal_views.py.
export const scoreColor = interpolateRgbBasis(["#f0efec", "#ec835a", "#d03b3b"]);
export const HIGHLIGHT = "#2a78d6";
