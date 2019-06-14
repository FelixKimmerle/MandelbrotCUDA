#include <cstdlib>

#include <iostream>
#include <string>

#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>
#include <cmath>
#include <iomanip>
#include <limits>
#include <vector>
#include <complex>
#include <gmpxx.h>
#include <thread>

#include "Kernel.h"

#include <sstream>

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

void DeleteTexture(GLuint &texture)
{
    if (texture != 0)
    {
        glDeleteTextures(1, &texture);
        texture = 0;
    }
}

void CreateTexture(GLuint &texture, unsigned int width, unsigned int height)
{
    // Make sure we don't already have a texture defined here
    DeleteTexture(texture);

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    // set basic parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Create texture data (4-component unsigned byte)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    // Unbind the texture
    glBindTexture(GL_TEXTURE_2D, 0);
}

void CreateCUDAResource(cudaGraphicsResource_t &cudaResource, GLuint GLtexture, cudaGraphicsMapFlags mapFlags)
{
    cudaGraphicsGLRegisterImage(&cudaResource, GLtexture, GL_TEXTURE_2D, mapFlags);
}

void DeleteCUDAResource(cudaGraphicsResource_t &cudaResource)
{
    if (cudaResource != 0)
    {
        cudaGraphicsUnregisterResource(cudaResource);
        cudaResource = 0;
    }
}

sf::Color interpolate(const sf::Color &color1, const sf::Color &color2,
                      const double &mid)
{
    int r = color1.r + (color2.r - color1.r) * mid;
    int g = color1.g + (color2.g - color1.g) * mid;
    int b = color1.b + (color2.b - color1.b) * mid;
    return sf::Color(r, g, b);
}

// This function takes a vector of colors, and returns a gradient based on
// the original vector
std::vector<sf::Color> color_table(const std::vector<sf::Color> &gradient)
{
    std::vector<sf::Color> v;
    double d = 0;
    for (auto i = gradient.begin(); i != gradient.end(); ++i)
    {
        // interpolate between this and the next color of the input gradient
        auto next = (i + 1 == gradient.end()) ? gradient.begin() : i + 1;
        for (double d = 0; d < 1; d += 0.01)
            v.push_back(interpolate(*i, *next, d));
    }
    return v;
}

sf::Color palette(const std::vector<sf::Color> &gradient, double zn_size, int iter)
{
    // use smooth coloring
    double nu = iter - std::log2(std::log2(zn_size));
    int i = (int)(nu * 10) % gradient.size();

    return gradient[i];
}

std::vector<std::complex<double>> deep_zoom_point(const mpf_class &center_r,
                                                  const mpf_class &center_i, int depth)
{
    std::vector<std::complex<double>> v;
    mpf_class xn_r = center_r;
    mpf_class xn_i = center_i;

    for (int i = 0; i != depth; ++i)
    {
        // pre multiply by two
        mpf_class re = xn_r + xn_r;
        mpf_class im = xn_i + xn_i;

        std::complex<double> c(re.get_d(), im.get_d());

        v.push_back(c);

        // make sure our numbers don't get too big
        //if (re > 1024 || im > 1024 || re < -1024 || im < -1024)
        //return v;

        // calculate next iteration, remember re = 2 * xn_r
        xn_r = xn_r * xn_r - xn_i * xn_i + center_r;
        xn_i = re * xn_i + center_i;
    }
    return v;
}

sf::Color pt(const int &i, const int &j, const std::vector<std::complex<double>> &x,
             const sf::Vector2u &size, const double &radius,
             const std::vector<sf::Color> &gradient)
{
    int window_radius = (size.x < size.y) ? size.x : size.y;
    // find the complex number at the center of this pixel
    std::complex<double> d0(radius * (2 * i - (int)size.x) / window_radius,
                            -radius * (2 * j - (int)size.y) / window_radius);

    int iter = 0;

    int max_iter = x.size();

    double zn_size;
    // run the iteration loop
    std::complex<double> dn = d0;
    do
    {
        dn *= x[iter] + dn;
        dn += d0;
        ++iter;
        zn_size = std::norm(x[iter] * 0.5 + dn);

        // use bailout radius of 256 for smooth coloring.
    } while (zn_size < 256 && iter < max_iter);

    // color according to iteration using logarithmic smoothing
    sf::Color color = palette(gradient, zn_size, iter);
    if (iter == max_iter)
        return sf::Color::Black; // if it's in the set, color black

    return color;
}

void updatepart(unsigned int startx, unsigned int endx, unsigned int starty, unsigned int endy, sf::VertexArray *set, const sf::Vector2u &size,
                const std::vector<std::complex<double>> &x, const double &radius,
                const std::vector<sf::Color> &gradient)
{
    for (unsigned int i = startx; i != endx; ++i)
    {
        for (unsigned int j = starty; j != endy; ++j)
        {
            (*set)[i + size.x * j].position = sf::Vector2f(i, j);
            (*set)[i + size.x * j].color = pt(i, j, x, size, radius, gradient);
        }
    }
}

void update(sf::VertexArray *set, const sf::Vector2u &size,
            const std::vector<std::complex<double>> &x, const double &radius,
            const std::vector<sf::Color> &gradient)
{
    // add a point for each pixel, coloring based on iteration
    unsigned concurentThreadsSupported = std::thread::hardware_concurrency();
    std::vector<std::thread *> t;
    for (unsigned int i = 0; i < concurentThreadsSupported; i++)
    {
        t.push_back(new std::thread(updatepart, i * (size.x / concurentThreadsSupported), (i + 1) * (size.x / concurentThreadsSupported), 0, size.y, set, size, x, radius, gradient));
    }
    for (unsigned int i = 0; i < concurentThreadsSupported; i++)
    {
        t[i]->join();
        delete t[i];
    }
    t.clear();
}

#define SIZE_X 1000
#define SIZE_Y 800
#define OFFSET 5

int main(int argc, char *argv[])
{
    Dimention<double> screen(0, SIZE_X, 0, SIZE_Y);
    Dimention<double> fract(-1.5, 1.5, -1.5, 1.5);

    sf::Font font;
    if (!font.loadFromFile("arial.ttf"))
    {
        std::cout << "Cant load Font" << std::endl;
    }
    sf::Text text;
    text.setFont(font);
    text.setCharacterSize(12);
    text.setFillColor(sf::Color(128, 128, 128));
    sf::RectangleShape rect;
    rect.setOutlineThickness(2);
    rect.setOutlineColor(sf::Color::Black);
    rect.setFillColor(sf::Color::White);

    GLuint g_GLPostprocessTexture = 0;
    cudaGraphicsResource_t g_CUDAGraphicsResource = 0;

    //cudaGLSetGLDevice(0);
    sf::RenderWindow window(sf::VideoMode(SIZE_X, SIZE_Y), "OpenGL", sf::Style::Default);
    window.setVerticalSyncEnabled(true);

    window.setActive(true);
    CreateTexture(g_GLPostprocessTexture, window.getSize().x, window.getSize().y);
    CreateCUDAResource(g_CUDAGraphicsResource, g_GLPostprocessTexture, cudaGraphicsMapFlagsWriteDiscard);
    glDisable(GL_DEPTH_TEST);

    sf::Vector2i startpos;
    bool Click = false;
    double m_Ddx = 0;
    double m_Ddy = 0;
    int max_iter = 100;
    bool running = true;
    bool dirty = true;
    int smooth = false;
    bool adaptive = false;
    float zoom = 0;
    bool fullscreen = false;
    bool vsync = true;
    bool GPU = true;
    bool infinite = false;
    bool showposition = false;
    bool animation = false;

    double destx;
    double desty;
    double step;
    bool relative;

    sf::Vector2u size = window.getSize();
    int pixels = size.x * size.y;
    sf::VertexArray *mandelbrot = new sf::VertexArray(sf::Points, pixels);

    // prepare gradient
    std::vector<sf::Color> gradient;
    gradient.push_back(sf::Color::Black);
    gradient.push_back(sf::Color::Blue);
    gradient.push_back(sf::Color(128, 0, 255));
    gradient.push_back(sf::Color::White);
    gradient.push_back(sf::Color::Yellow);
    gradient.push_back(sf::Color::Red);
    gradient = color_table(gradient);

    sf::Clock deltaclock;
    // default starting parameters
    double radius = 2;
    int depth = 1000;
    mpf_class center_r(0, 100);
    mpf_class center_i(0, 100);

    window.setVerticalSyncEnabled(vsync);
    while (running)
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
            {
                running = false;
            }
            else if (event.type == sf::Event::KeyPressed)
            {
                if (event.key.code == sf::Keyboard::Escape)
                {
                    running = false;
                }
                else if (event.key.code == sf::Keyboard::Add)
                {
                    max_iter += 10;
                    
                    dirty = true;
                    std::cout << "Max: " << max_iter << std::endl;
                }
                else if (event.key.code == sf::Keyboard::Subtract)
                {
                    max_iter -= 10;
                    if(max_iter <= 1)
                    {
                        max_iter = 1;
                    }
                    dirty = true;
                    std::cout << "Max: " << max_iter << std::endl;
                }
                else if (event.key.code == sf::Keyboard::Up)
                {
                    max_iter += 1;
                    dirty = true;
                    std::cout << "Max: " << max_iter << std::endl;
                }
                else if (event.key.code == sf::Keyboard::Down)
                {
                    max_iter -= 1;
                    if(max_iter <= 1)
                    {
                        max_iter = 1;
                    }
                    dirty = true;
                    std::cout << "Max: " << max_iter << std::endl;
                }
                else if (event.key.code == sf::Keyboard::Space)
                {
                    smooth++;
                    if (smooth >= 5)
                    {
                        smooth = 0;
                    }
                    std::cout << "Smooth Colors: " << smooth << std::endl;
                    dirty = true;
                }
                else if (event.key.code == sf::Keyboard::R)
                {
                    std::cout << "Reset" << std::endl;
                    double ratio = (double)window.getSize().x / (double)window.getSize().y;
                    zoom = 0;
                    max_iter = 100;
                    fract.reset(-1.5 * ratio, 1.5 * ratio, -1.5, 1.5);
                    dirty = true;
                }
                else if (event.key.code == sf::Keyboard::A)
                {
                    adaptive = !adaptive;
                    std::cout << "Adaptive iterations: " << (adaptive ? "ON" : "OFF") << std::endl;
                    if (adaptive)
                    {
                        max_iter = Dimention<float>::mmap(zoom, 0, 30, 100, 1100);
                        dirty = true;
                    }
                }
                else if (event.key.code == sf::Keyboard::P)
                {
                    showposition = !showposition;
                    std::cout << "Show position: " << (showposition ? "ON" : "OFF") << std::endl;
                }
                else if (event.key.code == sf::Keyboard::Z)
                {
                    if (!animation)
                    {
                        animation = true;
                        std::cout << "Input x y stepsize relative" << std::endl;
                        std::cin >> destx;
                        std::cin >> desty;
                        std::cin >> step;
                        std::cin >> relative;
                        std::cout << "Zoom started" << std::endl;
                        deltaclock.restart();
                        fract.reset(destx - fract.width()/2,(destx - fract.width()/2) + fract.width(),desty - fract.height()/2,(desty - fract.height()/2) + fract.height());

                    }
                    else
                    {
                        animation = false;
                    }
                }
                else if (event.key.code == sf::Keyboard::F11)
                {
                    fullscreen = !fullscreen;
                    std::cout << "Fullscreen: " << (fullscreen ? "ON" : "OFF") << std::endl;
                    if (fullscreen)
                    {
                        window.create(sf::VideoMode::getFullscreenModes()[0], "OpenGL", sf::Style::Fullscreen);
                        auto modes = sf::VideoMode::getFullscreenModes();
                        for (auto z = modes.begin(); z != modes.end(); z++)
                        {
                            std::cout << z->width << ":" << z->height << std::endl;
                        }
                    }
                    else
                    {
                        window.create(sf::VideoMode(SIZE_X, SIZE_Y), "OpenGL", sf::Style::Default);
                    }
                }
                else if (event.key.code == sf::Keyboard::V)
                {
                    vsync = !vsync;
                    window.setVerticalSyncEnabled(vsync);
                    std::cout << "VSync: " << (vsync ? "ON" : "OFF") << std::endl;
                }
                else if (event.key.code == sf::Keyboard::G)
                {
                    GPU = !GPU;
                    std::cout << "GPU: " << (GPU ? "ON" : "OFF") << std::endl;
                    dirty = true;
                }
                else if (event.key.code == sf::Keyboard::I)
                {
                    infinite = !infinite;
                    std::cout << "Infinite: " << (infinite ? "ON" : "OFF") << std::endl;
                    dirty = true;
                }
            }
            else if (event.type == sf::Event::Resized)
            {
                glViewport(0, 0, event.size.width, event.size.height);
                screen.reset(0, event.size.width, 0, event.size.height);
                double ratio = (double)window.getSize().x / (double)window.getSize().y;
                fract.reset(-1.5 * ratio, 1.5 * ratio, -1.5, 1.5);
                CreateTexture(g_GLPostprocessTexture, screen.width(), screen.height());
                DeleteCUDAResource(g_CUDAGraphicsResource);
                CreateCUDAResource(g_CUDAGraphicsResource, g_GLPostprocessTexture, cudaGraphicsMapFlagsWriteDiscard);
                window.setView(sf::View(sf::FloatRect(0,0,event.size.width,event.size.height)));
                dirty = true;
            }
            else if (event.type == sf::Event::MouseWheelScrolled)
            {
                sf::Vector2i mousexy = sf::Mouse::getPosition(window);

                double px = ((double)mousexy.x / (double)screen.width() * fract.width()) + fract.x_min();
                double py = ((double)mousexy.y / (double)screen.height() * fract.height()) + fract.y_min();

                double t = 0.1 * event.mouseWheelScroll.delta;
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
                {
                    t *= 3;
                }
                zoom += t;
                double xmax = std::abs(fract.x_max() - px);
                double xmin = std::abs(fract.x_min() - px);
                double ymax = std::abs(fract.y_max() - py);
                double ymin = std::abs(fract.y_min() - py);

                fract.reset(fract.x_min() + t * xmin, fract.x_max() - t * xmax, fract.y_min() + t * ymin, fract.y_max() - t * ymax);
                if (adaptive)
                {
                    max_iter = Dimention<float>::mmap(zoom, 0, 30, 100, 1100);
                }
                radius /= 1.1;
                dirty = true;
            }
        }

        if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left))
        {
            if (!Click)
            {
                startpos = sf::Mouse::getPosition(window);
            }

            sf::Vector2i mp = sf::Mouse::getPosition(window);
            if (mp.x >= 0 && mp.y >= 0)
            {
                double dx = (Dimention<double>::mmap(mp.x - startpos.x, screen.x_min(), screen.x_max(), fract.x_min(), fract.x_max()) - fract.x_min());
                double dy = (Dimention<double>::mmap(mp.y - startpos.y, screen.y_min(), screen.y_max(), fract.y_min(), fract.y_max()) - fract.y_min());

                fract.reset(fract.x_min() + m_Ddx - dx, fract.x_max() + m_Ddx - dx, fract.y_min() + m_Ddy - dy, fract.y_max() + m_Ddy - dy);
                dirty = true;
                m_Ddx = dx;
                m_Ddy = dy;
            }
            Click = true;
        }
        if (!sf::Mouse::isButtonPressed(sf::Mouse::Button::Left) && Click)
        {
            m_Ddx = 0;
            m_Ddy = 0;
            Click = false;
        }
        if (animation)
        {
            double xmax = std::abs(fract.x_max() - destx);
            double xmin = std::abs(fract.x_min() - destx);
            double ymax = std::abs(fract.y_max() - desty);
            double ymin = std::abs(fract.y_min() - desty);

            double t = step * deltaclock.restart().asSeconds();

            fract.reset(fract.x_min() + t * xmin, fract.x_max() - t * xmax, fract.y_min() + t * ymin, fract.y_max() - t * ymax);
            dirty = true;
        }
        if (dirty)
        {
            if (GPU)
            {
                glClear(GL_COLOR_BUFFER_BIT);

                if (infinite)
                {
                    center_r = (fract.x_max() + fract.x_min()) / 2.0;
                    center_i = (fract.y_max() + fract.y_min()) / 2.0;
                    radius = std::abs(fract.x_max() - fract.x_min()) / 3.75 * 2;
                    auto x = deep_zoom_point(center_r, center_i, max_iter);

                    std::vector<thrust::complex<double>> xgpu(x.size());
                    for (auto z = x.begin(); z != x.end(); z++)
                    {
                        xgpu.push_back(*z);
                    }

                    RenderA(g_CUDAGraphicsResource, screen, fract, max_iter, smooth, radius, x.size(), &xgpu[0]);
                }
                else
                {
                    Render(g_CUDAGraphicsResource, screen, fract, max_iter, smooth);
                }
            }
            else
            {
                center_r = (fract.x_max() + fract.x_min()) / 2.0;
                center_i = -(fract.y_max() + fract.y_min()) / 2.0;
                radius = std::abs(fract.x_max() - fract.x_min()) / 3.75 * 2;
                auto x = deep_zoom_point(center_r, center_i, max_iter);
                update(mandelbrot, size, x, radius, gradient);
            }
            //std::cout << center_r << " : " << center_i << std::endl;
            dirty = false;
        }
        if (GPU)
        {
            glBindTexture(GL_TEXTURE_2D, g_GLPostprocessTexture);
            glEnable(GL_TEXTURE_2D);

            glBegin(GL_QUADS);
            glTexCoord2f(0, 1);
            glVertex3f(-1, -1, 0);
            glTexCoord2f(1, 1);
            glVertex3f(1, -1, 0);
            glTexCoord2f(1, 0);
            glVertex3f(1, 1, 0);
            glTexCoord2f(0, 0);
            glVertex3f(-1, 1, 0);
            glEnd();

            glDisable(GL_TEXTURE_2D);
        }
        else
        {
            window.pushGLStates();
            window.clear();
            window.draw(*mandelbrot);
            window.popGLStates();
        }

        if (showposition)
        {
            window.pushGLStates();
            auto mpos = sf::Mouse::getPosition(window);
            double xpos = mpos.x / screen.width() * fract.width() + fract.x_min();
            double ypos = mpos.y / screen.height() * fract.height() + fract.y_min();
            text.setString("Real: " + to_string_with_precision(xpos, 20) + "\nImag: " + to_string_with_precision(ypos, 20) + "\nIter: " + std::to_string(max_iter));
            text.setPosition(mpos.x + OFFSET * 4, mpos.y + OFFSET);
            auto outrect = text.getGlobalBounds();
            rect.setPosition(outrect.left - OFFSET, outrect.top - OFFSET);
            rect.setSize(sf::Vector2f(outrect.width + OFFSET * 2, outrect.height + OFFSET * 2));
            window.draw(rect);
            window.draw(text);
            window.popGLStates();
        }
        window.display();
    }
    DeleteCUDAResource(g_CUDAGraphicsResource);
    DeleteTexture(g_GLPostprocessTexture);
    delete mandelbrot;
    std::cout << "Goodbye :)" << std::endl;
    return 0;
}
