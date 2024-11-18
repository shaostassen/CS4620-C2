
from PIL import Image as PIM
import requests, io
import numpy as np

try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

import matplotlib
import matplotlib.pyplot as plt

def aget_ipython():
    try:
        import IPython
        return IPython;
    except ImportError:
        print('Problem importing IPython');
        return None;

def runningInNotebook():
    try:
        ipyth = aget_ipython();
        if(ipyth.__class__.__name__ == 'module'):
            ipyth = ipyth.get_ipython();
        shell = ipyth.__class__.__name__;

        # shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


_ISNOTEBOOK = False;
if(runningInNotebook()):
    _ISNOTEBOOK = True;

def is_notebook():
    return _ISNOTEBOOK;
    # return runningInNotebook()

class Image(object):
    """Image
    """

    def __init__(self, path=None, pixels=None, convert_to_float=False, hdr=False, **kwargs):
        # You can do Image(pixels) or Image(path)
        self._samples = None;
        self.file_path = None;
        if (isinstance(path, np.ndarray) and (pixels is None)):
            # if the path looks like pixels and pixels are undefined, treat the path as pixels
            pixels = path;
        else:
            self.pixels = pixels;
            self.file_path = path;
            if(self.file_path is not None and pixels is None):
                self.loadImageData(self.file_path);

    def clone(self, share_data = False):
        selfclass = type(self);
        new_copy = selfclass(pixels=self.pixels.copy());
        new_copy.file_path = self.file_path;
        return new_copy;

    @property
    def pixels(self):
        """
        clipped pixel access
        :return:
        """
        return self.samples;

    @pixels.setter
    def pixels(self, data):
        self.samples = data;

    @property
    def n_color_channels(self):
        if (len(self.pixels.shape) < 3):
            return 1;
        else:
            return self.pixels.shape[2];

    @property
    def dtype(self):
        return self.pixels.dtype;

    @property
    def _is_float(self):
        return (self.dtype.kind in 'f');

    @property
    def _is_int(self):
        return (self.dtype.kind in 'iu');

    @property
    def fpixels(self):
        if (self._is_float):
            return self.pixels;
        else:
            return self.pixels.astype(float) * np.true_divide(1.0, 255.0);

    @property
    def _fpixels(self):
        if (self._is_float):
            return self.pixels;
        else:
            return self.pixels.astype(float) * np.true_divide(1.0, 255.0);

    @property
    def samples(self):
        return self._samples;

    @samples.setter
    def samples(self, value):
        self._samples = value;

    @property
    def ipixels(self):
        if (self._is_int):
            return self.pixels;
        else:
            return (self.pixels * 255).astype(np.uint8);

    @property
    def _ipixels(self):
        if (self._is_int):
            return self.pixels;
        else:
            return (self.pixels * 255).astype(np.uint8);

    @property
    def shape(self):
        return np.asarray(self.pixels.shape)[:];

    @property
    def _shape(self):
        return np.asarray(self.pixels.shape)[:];

    def pad(self, left=0, right=0, top=0, bottom=0, **kwargs):
        pad_width = [[top, bottom], [left, right], [0, 0]];
        self.pixels = np.pad(self.pixels, pad_width, **kwargs);
        return self;

    def GetPadded(self, left=0, right=0, top=0, bottom=0, **kwargs):
        rval = self.clone(share_data=False);
        rval.pad(left=left, right=right, top=top, bottom=bottom, **kwargs);
        return rval;

    def GetCropped(self, x_range=None, y_range=None):
        full_shape = self.shape;
        if (x_range is None):
            xrange = [0, full_shape[1]];
        else:
            xrange = [x_range[0], x_range[1]];
        if (y_range is None):
            yrange = [0, full_shape[0]];
        else:
            yrange = [y_range[0], y_range[1]];
        if (xrange[0] is None):
            xrange[0] = 0;
        if (xrange[1] is None):
            xrange[1] = full_shape[1];
        if (yrange[0] is None):
            yrange[0] = 0;
        if (yrange[1] is None):
            yrange[1] = full_shape[0];
        return Image(pixels=self.pixels[yrange[0]:yrange[1], xrange[0]:xrange[1]]);


    def setPixelTypeToFloat(self):
        if (self._is_float):
            return;
        else:
            self._samples = self._fpixels;
        return self;

    def setPixelTypeToUInt(self):
        if (self._is_int):
            return;
        else:
            self._samples = self._ipixels;
        return self;

    def GetUIntCopy(self):
        clone = self.clone(share_data=False);
        clone.setPixelTypeToUInt();
        return clone;

    def GetFloatCopy(self):
        clone = self.clone(share_data=False);
        clone.setPixelTypeToFloat();
        return clone;

    def clear(self):
        self.pixels = np.zeros(self.pixels.shape);


    def loadImageData(self, path=None, force_reload=True):
        if (path):
            self.file_path = path;
        if (self.file_path):
            if (force_reload or (not self.pixels)):
                pim = PIM.open(fp=self.file_path);
                self._samples = np.array(pim);

    @staticmethod
    def SolidRGBAPixels(shape, color=None):
        if (color is None):
            color = [0, 0, 0, 0];
        rblock = np.ones((shape[0], shape[1], 4));
        rblock[:] = color;
        return rblock;

    @staticmethod
    def SolidRGBPixels(shape, color=None):
        if (color is None):
            color = [0, 0, 0, 0];
        rblock = np.ones((shape[0], shape[1], 3));
        rblock[:] = color;
        return rblock;

    @classmethod
    def SolidImage(cls, shape, color=None):
        if (color is None):
            return cls(pixels=cls.SolidRGBAPixels(shape, [0, 0, 0]));
        elif (len(color) == 3):
            return cls(pixels=cls.SolidRGBPixels(shape, color));
        elif (len(color) == 4):
            return cls(pixels=cls.SolidRGBAPixels(shape, color));
        else:
            raise NotImplementedError;

    @classmethod
    def Zeros(cls, shape):
        return Image(pixels=np.zeros(shape));

    @classmethod
    def Ones(cls, shape):
        return Image(pixels=np.ones(shape));

    @classmethod
    def GaussianNoise(cls, size, mean=0, std=1):
        return cls(pixels=np.random.normal(mean, std, size));

    @property
    def width(self):
        return self.shape[1];

    @property
    def height(self):
        return self.shape[0];

    @property
    def possible_value_range(self):
        if (self.dtype.kind in 'iu'):
            return [0, 255];
        else:
            return [0.0, 1.0];

    def reflectY(self):
        self.pixels[:, :, :] = self.pixels[::-1, :, :];

    def reflectX(self):
        self.pixels[:, :, :] = self.pixels[:, ::-1, :];

    def PIL(self):
        return PIM.fromarray(np.uint8(self.ipixels));

    def _getRGBChannels(self):
        return self.pixels[:, :, 0:3];

    def GetRGBCopy(self, background=None):
        c = self.clone(share_data=False);
        if (c._colorspace is not Image.ColorSpaces.RGB):
            c._converColorSpaceToRGB();
        if (c.n_color_channels == 1):
            c._samples = np.stack((c._samples,) * 3, axis=-1);
            return c;
        if (c.n_color_channels == 4):
            bg = np.zeros_like((self.shape[0], self.shape[1], 3));
            if (background is not None):
                bg[:] = background;
            alpha = self.fpixels[:, :, 3];
            a = np.dstack([alpha, alpha, alpha]);
            c._samples = bg * (1.0 - a) + a * c.fpixels[:, :, :3];
            return c;
        return c;

    def _addChannelToUnclippedSamples(self, channel_data):
        if (channel_data.dtype.kind == self.dtype.kind):
            self._samples = np.dstack((self._samples, channel_data));
            return;

        if (channel_data.dtype.kind in 'iu'):
            if (self._is_int):
                self._samples = np.dstack((self._samples, channel_data));
                return;
            else:
                self._samples = np.dstack((self._samples, channel_data.astype(float) * np.true_divide(1.0, 255.0)));
                return;
        else:
            assert (channel_data.dtype.kind in 'f'), "unknown dtype {}".format(channel_data.dtype);
            if (self._is_float):
                self._samples = np.dstack((self._samples, channel_data));
                return;
            else:
                self._samples = np.dstack((self._samples, (channel_data * 255).astype(np.uint8)));
                return;
        assert (False), "Should not get here!\nchannel_data.dtype={}\nself.pixels.dtype={}".format(channel_data.dtype,
                                                                                                   self.pixels.dtype);

    def GetRGBACopy(self):
        if (self.n_color_channels == 4):
            return self.clone(share_data=False);
        if (self.n_color_channels == 3):
            clone = self.clone(share_data=False);
            clone._addChannelToUnclippedSamples(np.ones(self._samples.shape[:2]));
            return clone;

    def GetGrayCopy(self):
        if (self.n_color_channels == 1):
            return self.clone(share_data=False);
        if (self.n_color_channels == 4):
            return self.GetRGBCopy().GetGrayCopy();
        if (self.n_color_channels == 3):
            clone = self.clone(share_data=False);
            clone.pixels = np.mean(clone.fpixels, 2);
            return clone;

    def GetFFTImage(self):
        fftim = Image.Zeros(self.shape);
        if (self.n_color_channels == 1):
            fftim.pixels = np.fft.fftshift(np.fft.fft2(self.fpixels));
        else:
            for c in range(self.n_color_channels):
                fftim.pixels[:, :, c] = np.fft.fftshift(np.fft.fft2(self.fpixels[:, :, c]));
        return fftim;

    def normalize(self, scale=None):
        if (scale is None):
            scale = self.valrange[1];

        self.pixels = self.pixels / np.max(self.pixels.ravel());
        self.pixels = self.pixels * scale;

    def show(self, title=None, new_figure=True, **kwargs):
        if (is_notebook()):
            Image.Show(self, new_figure=new_figure, title=title, **kwargs);
        else:
            self.PIL().show();

    def showChannel(self, channel=0, **kwargs):
        cim = self.getChannelImage(channel);
        cim.show();

    def getChannelImage(self, channel=0, **kwargs):
        return Image(pixels=self._getChannelPixels(channel));

    def _setValueRange(self, value_range=None):
        if (value_range is None):
            value_range = [0, 1];
        data = self.samples;
        maxval = np.max(data);
        minval = np.min(data)
        currentscale = maxval - minval;
        data = data - minval;
        data = data * (value_range[1] - value_range[0]) / currentscale
        data = data + value_range[0];
        self._samples = data;

    def GetWithValuesMappedToRange(self, value_range=None):
        if (value_range is None):
            value_range = [0, 1];
        remap = self.clone();
        remap._setValueRange(value_range=value_range);
        return remap;

    @staticmethod
    def Show(im, title=None, new_figure=True, axis=None, **kwargs):
        if (isinstance(im, Image)):
            imdata = im.pixels;
        else:
            imdata = im;

        if (imdata.dtype == np.int64 or imdata.dtype == np.int32):
            imdata = imdata.astype(np.uint8)

        if (is_notebook()):
            if (new_figure):
                if (title is not None):
                    f = plt.figure(num=title);
                else:
                    f = plt.figure();
            if (len(imdata.shape) < 3):
                if (imdata.dtype == np.uint8):
                    nrm = matplotlib.colors.Normalize(vmin=0, vmax=255);
                else:
                    nrm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0);
                if (axis is not None):
                    axis.imshow(imdata, cmap='gray', norm=nrm, **kwargs);
                else:
                    plt.imshow(imdata, cmap='gray', norm=nrm, **kwargs);


            elif (imdata.shape[2] == 2):
                if (axis is not None):
                    axis.imshow(Image._Flow2RGB(imdata), **kwargs);
                else:
                    plt.imshow(Image._Flow2RGB(imdata), **kwargs);
            else:
                if (axis is not None):
                    axis.imshow(imdata, **kwargs);  # divided by 255
                else:
                    plt.imshow(imdata, **kwargs);  # divided by 255
            plt.axis('off');
            if (title):
                plt.title(title);

    def _getPlayHTML(self, format='png'):
        a = np.uint8(self.samples);
        f = StringIO();
        PIM.fromarray(a).save(f, format)
        ipdob = aget_ipython().display.Image(data=f.getvalue());
        # encoded = base64.b64encode(f.getvalue());
        imghtml = '''<img src="data:image/png;base64,{0}"'''.format(ipdob._repr_png_);
        return imghtml;
        # return ipdob._repr_html_();

    def play(self):
        """
        just to be compatible with other MediaObjects...
        :return:
        """
        self.show();

    def GetWithTaperedAlphaBoundary(self, border_width=None):
        if (border_width == 'default'):
            border_width = int(min(self.width, self.height) * Image._DEFAULT_BORDER_THICKNESS);
        rval = self.GetRGBACopy().GetFloatCopy();
        rval.pixels[:border_width, :, 3] = np.transpose(np.tile(np.linspace(0, 1, border_width), (rval.width, 1)))
        rval.pixels[-border_width:, :, 3] = np.transpose(np.tile(np.linspace(1, 0, border_width), (rval.width, 1)))
        rval.pixels[:, :border_width, 3] = np.tile(np.linspace(0, 1, border_width), (rval.height, 1)) * rval.pixels[:,
                                                                                                        :border_width,
                                                                                                        3]
        rval.pixels[:, -border_width:, 3] = np.tile(np.linspace(1, 0, border_width), (rval.height, 1)) * rval.pixels[:,
                                                                                                         -border_width:,
                                                                                                         3]
        return rval

    def GetTaperedBorderAlpha(self, border_width=None):
        w = self.width;
        h = self.height;
        # print("Border width: {}".format(border_width))
        if (border_width == 'default'):
            border_width = int(min(w, h) * Image._DEFAULT_BORDER_THICKNESS);
            # print("using border width {}".format(border_width));
        rval = np.ones([self.height, self.width]);
        rval[:border_width, :] = np.transpose(np.tile(np.linspace(0, 1, border_width), (w, 1)))
        rval[-border_width:, :] = np.transpose(np.tile(np.linspace(1, 0, border_width), (w, 1)))
        rval[:, :border_width] = np.tile(np.linspace(0, 1, border_width), (h, 1)) * rval[:, :border_width]
        rval[:, -border_width:] = np.tile(np.linspace(1, 0, border_width), (h, 1)) * rval[:, -border_width:]
        return rval

    @classmethod
    def FromURL(cls, url):
        response = requests.get(url)
        bytes_im = io.BytesIO(response.content)
        pix = np.array(PIM.open(bytes_im));
        im = cls(pixels=pix);
        return im;

    @classmethod
    def FromPlotFig(cls, fig, shape=None):
        if (shape is not None):
            fig.set_size_inches(shape[1] / fig.dpi, shape[0] / fig.dpi);
        fig.canvas.draw();
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='');
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,));
        return Image(pixels=data);

    def writeToFile(self, output_path=None, **kwargs):
        self.PIL().save(output_path);

    def _getChannelPixels(self, channel=None):
        if (channel is None):
            channel = 0;
        return self.pixels[:, :, channel];

    def _getAlphaPixels(self):
        if (self.n_color_channels < 4):
            if (self._is_float):
                return np.ones([self.imshape[0], self.imshape[1]]);
            else:
                return (np.ones([self.imshape[0], self.imshape[1]]) * 255).astype(np.uint8);
        else:
            return self._getChannelPixels(3);

    def GetAlphaAsRGB(self):
        alpha = self._getAlphaPixels();
        return Image(pixels=np.dstack([alpha, alpha, alpha]));
        # return Image._MultiplyArrayAlongChannelDimension(alpha, 3);

    @classmethod
    def FourierBasis(cls, n_vectors=100):
        C = np.zeros((n_vectors, n_vectors, 3));
        ns = np.arange(n_vectors)
        one_cycle = 2 * np.pi * ns / n_vectors
        for k in range(n_vectors):
            t_k = k * one_cycle
            C[k, :, 0] = np.cos(t_k)
            C[k, :, 1] = np.sin(t_k)
            C[k, :, 2] = 0
        return Image(pixels=C);

    @classmethod
    def GetCoordIm(cls, size, normalized=True):
        if (normalized):
            y = np.linspace(0, 1, size[0]);
            x = np.linspace(0, 1, size[1]);
        else:
            y = np.arange(size[0]);
            x = np.arange(size[1]);
        yy, xx = np.meshgrid(y, x)
        return Image(pixels=np.dstack((yy, xx)));

    def getCoordinateIm(self, normalized=True):
        return Image.GetCoordIm(self.shape, normalized=normalized);

    def Plot3D(self):
        xx, yy = np.mgrid[0:self.shape[0], 0:self.shape[1]]
        # create the figure
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(xx, yy, self.pixels, rstride=1, cstride=1, cmap=plt.cm.gray,
                        linewidth=0)
        # show it
        plt.show()

    def PlotBrightnessHistogram(self, nbins=100, colormap='rainbow'):
        # import matplotlib.pyplot as plt
        # plt.imshow(imV.pixels, colormap='jet')

        # colormap='jet'
        # colormap='coolwarm'
        hsv = self.GetHSV();
        fig = plt.figure()
        vpix = hsv.pixels[:, :, 2];
        pos = plt.imshow(vpix, cmap=colormap)
        fig.colorbar(pos)
        plt.show()
        data = vpix.ravel();
        cm = plt.cm.get_cmap(colormap)
        # Plot histogram.
        n, bins, patches = plt.hist(data, nbins)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        # scale values to interval [0,1]
        col = bin_centers - min(bin_centers)
        col /= max(col)
        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))
        plt.show()

    @classmethod
    def CreateFromMatPlotLibFigure(cls, fig):
        fig.canvas.draw();
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='');
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,));
        return cls(pixels=data);

    @classmethod
    def StackImages(cls, images, concatdim=0, **kwargs):
        matchdim = (concatdim + 1) % 2;
        newframe = images[0].clone(share_data=False).pixels;
        for vn in range(1, len(images)):
            addpart = images[vn].pixels;
            partsize = np.asarray(addpart.shape)[:];
            cumulsize = np.asarray(newframe.shape)[:];
            if (partsize[matchdim] != cumulsize[matchdim]):
                sz = partsize[:];
                sz[matchdim] = cumulsize[matchdim];
                addpart = np.array(PIM.fromarray(addpart).resize((int(sz[1]), int(sz[0]))));
            newframe = np.concatenate((newframe, addpart), concatdim);
        return cls(pixels=newframe)

    @classmethod
    def CreateCheckerImage(cls, size=None, grid_size=None, col1=None, col2=None):
        if (size is None):
            size = [256, 256, 3];
        w = size[1];
        h = size[0];
        if (col1 is None):
            col1 = np.ones(3);
        if (col2 is None):
            col2 = np.zeros(3);
        im = Image(pixels=np.zeros(size).astype(float));
        if (grid_size is None):
            grid_size = int(min(w, h) / 25);
        # Make pixels white where (row+col) is odd
        for i in range(w):
            for j in range(w):
                if (i // grid_size + j // grid_size) % 2:
                    im.pixels[i, j] = col1
                else:
                    im.pixels[i, j] = col2
        return im;

    ##################//--operators--\\##################
    # <editor-fold desc="operators">
    def __add__(self, other):
        if (isinstance(other, self.__class__)):
            return self._selfclass(pixels=np.add(self.samples, other.samples));
        else:
            return self._selfclass(pixels=np.add(self.samples, other));

    def __radd__(self, other):
        return self.__add__(other);

    def __sub__(self, other):
        if (isinstance(other, self.__class__)):
            return self._selfclass(pixels=np.subtract(self.samples, other.samples));
        else:
            return self._selfclass(pixels=np.subtract(self.samples, other));

    def __rsub__(self, other):
        if (isinstance(other, (NDArray))):
            return self._selfclass(pixels=np.subtract(other._ndarray, self._ndarray));
        elif (isinstance(other, self.__class__)):
            return self._selfclass(pixels=np.subtract(other.pixels, self.pixels));
        else:
            return self._selfclass(pixels=np.subtract(other, self.pixels));

    def __mul__(self, other):
        if (isinstance(other, self.__class__)):
            return self._selfclass(pixels=np.multiply(self.samples, other.samples));
        else:
            return self._selfclass(pixels=np.multiply(self.samples, other));

    def __rmul__(self, other):
        return self.__mul__(other);

    # </editor-fold>
    ##################\\--operators--//##################

    @staticmethod
    def imshow(im, new_figure=True):
        if (isinstance(im, Image)):
            imdata = im.pixels;
        else:
            imdata = im;

        if (imdata.dtype == np.int64 or imdata.dtype == np.int32):
            imdata = imdata.astype(np.uint8)

        if (is_notebook()):
            if (new_figure):
                plt.figure();
            if (len(imdata.shape) < 3):
                if (imdata.dtype == np.uint8):
                    nrm = matplotlib.colors.Normalize(vmin=0, vmax=255);
                else:
                    nrm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0);
                plt.imshow(imdata, cmap='gray', norm=nrm);
            else:
                plt.imshow(imdata);  # divided by 255
            plt.axis('off');
    # </editor-fold>
    ##################\\--Image Text--//##################


##################//--Decorators--\\##################
# <editor-fold desc="Decorators">
def ImageMethod(func):
    setattr(Image, func.__name__, func)
    return getattr(Image, func.__name__);


def ImageStaticMethod(func):
    setattr(Image, func.__name__, staticmethod(func))
    return getattr(Image, func.__name__);


def ImageClassMethod(func):
    setattr(Image, func.__name__, classmethod(func))
    return getattr(Image, func.__name__);


# </editor-fold>
##################\\--Decorators--//##################


# url = "https://www.cs.cornell.edu/courses/cs4620/2023fa/cs4620-web/images/LabCatFloatingHead.png"
# im = Image.FromURL(url)
# im.show()
