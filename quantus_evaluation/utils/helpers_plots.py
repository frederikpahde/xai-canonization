import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from zennit import image as zimage
from PIL import Image, ImageDraw, ImageFont
from utils.helpers import compute_aopc
import re

def get_img_lrp_flat(image_lrp):
    amax = image_lrp.max((0, 1), keepdims=True)
    image_lrp = (image_lrp + amax) / 2 / amax
    return image_lrp
    
def plot_attributions(x_batch, attributions, denormalizer, save_dir):
    num_samples = min(8, len(x_batch))
    size = 3
    num_rows = len(attributions) + 1
    ncols = num_samples
    fig, axes = plt.subplots(nrows=num_rows, ncols=ncols, figsize=(ncols * size, num_rows * size), squeeze=False)

    for index in range(num_samples):
        sample = x_batch[index]
        if denormalizer:
            sample = denormalizer(sample)
        img_in = np.moveaxis(sample.cpu().detach().numpy(), 0, 2)
        axes[0][index].imshow(img_in)
        axes[0][index].axis('off')

        for j, (explainer_name, attr) in enumerate(attributions.items()):
            
            img_lrp = zimage.imgify(get_img_lrp_flat(attr[index].squeeze()), vmin=0, vmax=1., level=2.0, cmap='coldnhot')
            axes[j + 1][index].imshow(img_lrp)
            axes[j + 1][index].axis('off')
            
            axes[j + 1][0].set_title(explainer_name)
        
        
    axes[0][0].set_title("Input Image")
    plt.subplots_adjust(wspace=.0, hspace=.2)

    for filetype in ["png", "pdf"]:
        path_fig = f"{save_dir}/sample_explanations.{filetype}"
        fig.savefig(path_fig, bbox_inches='tight')

def plot_attributions_clevr(batch, attributions, pred_batch, save_dir, vocab_q, inv_vocab_a):
    num_samples = min(8, len(batch['qid']))
    size = 3
    num_rows = len(attributions) + 2
    ncols = max(2, num_samples)
    fig, axes = plt.subplots(nrows=num_rows, ncols=ncols, figsize=(ncols * size, num_rows * size))

    for index in range(num_samples):
        sample = batch['image'][index]
        img_in = np.moveaxis(sample.cpu().detach().numpy(), 0, 2)
        axes[0][index].imshow(img_in)
        axes[0][index].axis('off')
        axes[1][index].axis('off')

        for j, (explainer_name, attr) in enumerate(attributions.items()):
            img_lrp = zimage.imgify(get_img_lrp_flat(attr[index].squeeze()), vmin=0, vmax=1., level=4.0, cmap='bwr')
            axes[j + 2][index].imshow(img_lrp)
            axes[j + 2][index].axis('off')
            
            axes[j + 2][0].set_title(explainer_name)
        
        
    axes[0][0].set_title("Input Image")
    plt.subplots_adjust(wspace=.0, hspace=.2)

    for filetype in ["png", "pdf"]:
        path_fig = f"{save_dir}/sample_explanations.{filetype}"
        fig.savefig(path_fig)


def plot_single_pixel_flipping_experiment(y_batch, scores, class_name_by_index, ax):
    
    aopc_sum = None
    num_samples = 0
    for c in np.unique(y_batch):
        indices = np.where(y_batch == c)
        aopc = compute_aopc(np.array(scores)[indices])
        ax.plot(
            np.linspace(0, 1, len(scores[0])),
            np.mean(np.array(scores)[indices], axis=0),
            label=f"({aopc:.3f}) target: {class_name_by_index[c]} ({indices[0].size} samples)",
        )
        aopc_sum = aopc if aopc_sum is None else aopc_sum + aopc
        num_samples += len(indices)
        
    ax.set_xlabel("Fraction of pixels flipped")
    ax.set_ylabel("Mean Prediction")
    ax.set_ylim([0, 1])
    ax.set_yticklabels(["{:.0f}%".format(x * 100) for x in ax.get_yticks()])
    ax.set_xticklabels(["{:.0f}%".format(x * 100) for x in ax.get_xticks()])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    return aopc_sum/num_samples

def aggregate_and_plot_pixel_flipping_experiments(results, y_all, metric_name, metric_name_new, class_name_by_index, save_dir):
    
    n_rows = int(np.ceil(len(results.keys()) / 2))
    fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(18, 5 * n_rows), squeeze=False)

    for i, (key, scores) in enumerate(results.items()):
        apoc_avg = plot_single_pixel_flipping_experiment(y_batch=y_all, 
                                                         scores=scores[metric_name], 
                                                         class_name_by_index=class_name_by_index,
                                                         ax=axes[i // 2][i % 2])
        axes[i // 2][i % 2].set_title(f"{metric_name} - {key} (Avg. AOPC: {apoc_avg:.3f})")
        results[key][metric_name_new] = apoc_avg

    if len(results.keys()) % 2 != 0:
        axes[-1][-1].axis('off')
        
    plt.subplots_adjust(wspace=1, hspace=.4)

    metric_fname = metric_name
    for c in ["(", ")", " "]:
        metric_fname = metric_fname.replace(c, "_")
        
    for filetype in ["png", "pdf"]:
        path_fig = f"{save_dir}/{metric_fname}.{filetype}"
        fig.savefig(path_fig, bbox_inches='tight')
    plt.close()
    return results

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default."""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default."""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels, angles=None):
            self.set_thetagrids(angles=np.degrees(theta), labels=labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped."""
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)


        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))

                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)

                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

COLORS = ["#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
"#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
"#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
"#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
"#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
"#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
"#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
"#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
"#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
"#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
"#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
"#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
"#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C"]

def plot_spyder_graph(df_ranks, save_dir):
    # Make spyder graph!
    data = [df_ranks.columns.values, (df_ranks.to_numpy())]
    theta = radar_factory(len(data[0]), frame='polygon')
    spoke_labels = data.pop(0)

    fig, ax = plt.subplots(figsize=(11, 11), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)
    for i, (d, method) in enumerate(zip(data[0], list(df_ranks.index))):
        line = ax.plot(theta, d, label=method, color=COLORS[i], linewidth=5.0)
        ax.fill(theta, d, alpha=0.15)

    # Set lables.
    ax.set_varlabels(labels=list(df_ranks.columns))
    ax.set_rgrids(np.arange(0, df_ranks.values.max() + 0.5), labels=[]) 

    # Put a legend to the right of the current axis.
    ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5))

    for filetype in ["png", "pdf"]:
        results_spyder_path = f"{save_dir}/spyder_graph.{filetype}"
        fig.savefig(results_spyder_path, bbox_inches='tight')
    plt.close()


def imgify_text(batch, index, pred_batch, vocab_q, inv_vocab_a, 
                vmin=0.0, vmax=1.0):
    path_relevance_text = batch['path_rel_text_precomputed'][index]
    rel_text = np.load(path_relevance_text)
    
    rel_text = rel_text.sum(axis=1).reshape(1, -1)
    rel_text = get_img_lrp_flat(rel_text)
    
    rel_text = (rel_text - vmin) / (vmax - vmin)
    rel_text = (rel_text * 255).clip(0, 255).astype(np.uint8)
    colors = zimage.palette()[rel_text][0] 
    return draw_text(batch, index, colors, pred_batch, vocab_q, inv_vocab_a)

def clean_word(word):
    for i in [r'\?',r'\!',r'\-',r'\'',r'\"',r'\$',r'\:',r'\@',r'\(',r'\)',r'\/',r'\,',r'\.',r'\;']: # remove all punctuation
        word = re.sub( i, '', word)

    # punctuation should be separated from the words
    word = re.sub('([.,;:!?()])', r' \1 ', word)
    word = re.sub('\s{2,}', ' ', word)
    return word.lower()

def draw_text(batch, index, colors, pred_batch, vocab_q, inv_vocab_a):
    sentence = batch['question_text'][index]
    image = Image.new("RGB", (256, 256), "white")
    fontsize = 18
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fontsize)

    draw = ImageDraw.Draw(image)

    offset = 5
    start_x = 5
    start_y = 25
    
    counter = 0
    for i, w in enumerate(sentence.split()):
        word_clean = clean_word(w)
        position = (start_x + offset, start_y + offset)
        text=w
        bbox = draw.textbbox(position, text, font=font)
        if bbox[2] > 200:
            start_x = 5
            start_y = bbox[3]
        else:
            start_x = bbox[2]

        if word_clean in list(vocab_q.keys()):
            draw.rectangle(bbox, fill=tuple(colors[i]))
            counter += 1
        draw.text(position, text, fill="black", font=font)

    ## Response
    text_correct = f"Correct: {inv_vocab_a[batch['answer'][index].item()]}"
    text_predict = f"Predict: {inv_vocab_a[pred_batch[index].item()]}"

    draw.text((20, 100), text_correct, fill="black", font=font)
    draw.text((20, 120), text_predict, fill="black", font=font)
    
    return image