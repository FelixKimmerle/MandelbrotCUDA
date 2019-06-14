#pragma once
#include <SFML/System.hpp>
template <typename T>
class Dimention
{
	T _x_min, _x_max, _y_min, _y_max;

  public:
	Dimention(T x_min, T x_max, T y_min, T y_max)
		: _x_min(x_min), _x_max(x_max), _y_min(y_min), _y_max(y_max)
	{
	}

	// Utility functions for getting the size, width and height of the window
	T size() const
	{
		return (width() * height());
	}

	T width() const
	{
		return (_x_max - _x_min);
	}

	T height() const
	{
		return (_y_max - _y_min);
	}

	// Getters and setters for the window elements
	T x_min() const
	{
		return _x_min;
	}

	void x_min(T x_min)
	{
		_x_min = x_min;
	}

	T x_max() const
	{
		return _x_max;
	}

	void x_max(T x_max)
	{
		_x_max = x_max;
	}

	T y_min() const
	{
		return _y_min;
	}

	void y_min(T y_min)
	{
		_y_min = y_min;
	}

	T y_max() const
	{
		return _y_max;
	}

	void y_max(T y_max)
	{
		_y_max = y_max;
	}

	// Reset all values
	void reset(T x_min, T x_max, T y_min, T y_max)
	{
		_x_min = x_min;
		_x_max = x_max;
		_y_min = y_min;
		_y_max = y_max;
	}
	static inline T mmap(T value, T start1, T stop1, T start2, T stop2)
	{
		return start2 + (stop2 - start2) * ((value - start1) / (stop1 - start1));
	}

	sf::Vector2<T> Scale(sf::Vector2<T> &point, Dimention &d)
	{
		T X = mmap(point.x, d.x_min(), d.x_max(), this->x_min(), this->x_max());
		T Y = mmap(point.y, d.y_min(), d.y_max(), this->y_min(), this->y_max());

		return sf::Vector2<T>(X, Y);
	}
};