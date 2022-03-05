from reprep import Report
from zuper_commons.text import remove_escapes


def generate_homotopy_report(homotopies_sorted):
    r_eva = Report('Homotopy_evaluation')
    texts = []
    for homotopy in homotopies_sorted:
        texts.append(
            f"\t{homotopy.homo_class}: \n"
            f"\theuristic(shortest time)={homotopy.heuristic},\n"
        )
    text = "\n".join(texts)
    r_eva.text("Evaluation", remove_escapes(text))
    return r_eva
