import pandoc
from pdf2image import convert_from_bytes
import numpy as np

FONTS=[None, 'Arial', 'Times New Roman']

def _markdown_cfg(font=None):
    return \
r"""geometry:
- top=2.1cm
- bottom=2.1cm
- left=2.97cm
- right=2.97cm
header-includes:
- \usepackage{multicol}
- \newcommand{\hideFromPandoc}[1]{#1}
- \hideFromPandoc{\let\Begin\begin\let\End\end}
- \pagenumbering{gobble}""" + \
(f"\n- \\setmainfont[]{{{font}}}" if font else '')


def _ingredients_basic(ingredients: list[str]):
    return ''.join([f"- {ingredient}\n" for ingredient in ingredients])


def _instructions_basic(instructions: list[str]):
    return ''.join([f"{i+1}. {instruction}\n" for i, instruction in enumerate(instructions)])


def basic_template(recipe_name, ingredients: list[str], instructions: list[str], font=None):
    return f"""
---
{_markdown_cfg(font)}
---
# {recipe_name}

## Ingredients
{_ingredients_basic(ingredients)}

## Instructions
{_instructions_basic(instructions)}
"""


def two_column_template(recipe_name, ingredients: list[str], instructions: list[str], font=None):
    return f"""
---
{_markdown_cfg(font)}
---
# {recipe_name}
\\Begin{{multicols}}{{2}}
## Ingredients
{_ingredients_basic(ingredients)}
## Instructions
{_instructions_basic(instructions)}
\\End{{multicols}}
"""


TEMPLATES = [basic_template, two_column_template]


def convert_to_image(input_text: str, format = 'markdown'):
    doc = pandoc.read(input_text, format=format)
    pdf = pandoc.write(doc, format='pdf', options=['--pdf-engine=lualatex'])
    images = convert_from_bytes(pdf)

    return images


def show_image(images):
    import matplotlib.pyplot as plt
    from PIL import Image
    if type(images) is list:
        images = np.hstack(images)
    plt.imshow(Image.fromarray(images))
    plt.show()

def main():
    # Changing fonts
    # arev, times
    text = \
r"""
---
fontfamily: times
header-includes:
- \usepackage{multicol}
- \newcommand{\hideFromPandoc}[1]{#1}
- \hideFromPandoc{
    \let\Begin\begin
    \let\End\end
  }
---

# recipe

\Begin{multicols}{2}

## Ingredients
- 2 cups of flour
- 1 cup of sugar
- 1/2 cup of butter
- 1 cup of milk
- 2 eggs

$$\sum_{x=0}^n x$$

## Instructions
1. Preheat the oven to 350°F (175°C).
2. In a bowl, mix the flour, sugar, and butter.
3. Add the milk and eggs, and mix until smooth.
4. Pour the mixture into a greased baking dish.
5. Bake for 30 minutes or until golden brown.
6. Longer
7. Let it cool before serving.
8. Enjoy your delicious recipe!
9. Longer
10. Longer
11. Longer
12. Longer
13. Longer

\End{multicols}

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus posuere, neque fermentum tristique sagittis, ipsum ipsum condimentum dui, sed viverra libero velit eu mi. Cras id porttitor orci. Nulla ut rhoncus neque. Pellentesque et orci eu sem ullamcorper varius. Nunc non turpis vel odio tempus finibus ut ac justo. Duis ac dolor ipsum. Donec fermentum feugiat cursus.

In egestas lacus nec nulla imperdiet tincidunt. Pellentesque nec elementum lorem. Sed vestibulum nulla iaculis erat semper, vitae pretium erat semper. Nunc porttitor dui a leo sollicitudin convallis. Aenean ac consequat justo. Proin non bibendum libero. Quisque sit amet ex elit. Praesent vitae eros ultrices metus consequat eleifend. Interdum et malesuada fames ac ante ipsum primis in faucibus. Pellentesque non auctor sapien. Sed nec justo ac augue dignissim pretium a at mauris. In quis magna id dui luctus consectetur. Phasellus sit amet tellus id elit elementum faucibus nec quis justo. Donec pellentesque ac lectus ac ultricies. Donec vitae felis sagittis, aliquet urna auctor, sollicitudin nunc. Integer ut mattis justo.

Cras venenatis nisl a ligula suscipit, a varius libero faucibus. Vivamus semper augue eget ipsum condimentum mollis. Pellentesque eget dui massa. Pellentesque cursus tincidunt nulla, vel lobortis sapien posuere vitae. Suspendisse potenti. Mauris interdum volutpat dictum. Sed efficitur, lectus nec suscipit luctus, leo elit lacinia velit, et rhoncus diam ipsum rhoncus justo. Fusce maximus quam a fermentum aliquet. Sed mollis eget odio sed imperdiet. Maecenas nisi neque, egestas a elit et, porttitor placerat diam. Ut risus dui, fringilla ultrices vestibulum a, porttitor vel est. Ut sed rhoncus sapien, vitae mattis erat. Donec condimentum justo felis, non congue nunc accumsan nec. Phasellus neque neque, ultrices sit amet vestibulum sed, porta ut mauris. Donec lorem orci, hendrerit nec pretium sed, luctus ut dui. Sed laoreet tortor quis nunc pellentesque lacinia.

Morbi finibus eleifend mi a egestas. Etiam pellentesque tincidunt ex, vestibulum pretium purus sagittis fringilla. Suspendisse sit amet suscipit libero. Nam ornare feugiat velit, et aliquet ex aliquet in. Ut magna arcu, sagittis non eleifend nec, bibendum vel lorem. Proin consequat odio consequat odio fermentum, ut auctor nunc finibus. Pellentesque vel nulla justo. Vivamus id ante sem. Vestibulum pellentesque at eros ut ullamcorper. Vestibulum faucibus enim vel posuere dapibus. Integer vel finibus metus.

Quisque nec velit sagittis tellus mollis pulvinar et ac elit. Mauris at erat suscipit, gravida orci ut, suscipit lorem. Quisque in dictum metus, in congue dui. Nullam venenatis arcu erat, ut luctus enim maximus ac. Aenean ullamcorper accumsan magna non rutrum. Ut interdum sem ligula, vel tristique odio auctor eu. Cras lobortis nulla vel sem mollis, egestas eleifend leo dapibus. Donec eros purus, varius vitae neque non, accumsan mollis urna. Quisque pulvinar mi fermentum ullamcorper auctor. 
"""

    text = two_column_template("recipe", ["2 cups of flour", "1 cup of sugar", "1/2 cup of butter", "1 cup of milk", "2 eggs"], ["Preheat the oven to 350°F (175°C).", "In a bowl, mix the flour, sugar, and butter.", "Add the milk and eggs, and mix until smooth.", "Pour the mixture into a greased baking dish.", "Bake for 30 minutes or until golden brown.", "Let it cool before serving.", "Enjoy your delicious recipe!"], font=FONTS[1])
    images = convert_to_image(text)
    show_image(images)


if __name__ == "__main__":
    main()