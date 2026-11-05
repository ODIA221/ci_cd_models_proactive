// Libellés de l'interface : français par défaut, anglais avec ?lang=en (figures
// des articles en anglais). ?figure=1 n'affiche que les trois vues, pour une
// capture reproductible (voir docs/07).
const params = new URLSearchParams(window.location.search);
export const isEnglish = params.get("lang") === "en";
export const isFigure = params.get("figure") === "1";
export const tr = (fr: string, en: string) => (isEnglish ? en : fr);
