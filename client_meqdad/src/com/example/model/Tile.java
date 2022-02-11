package com.example.model;

import java.util.Objects;

public class Tile {
    /*
        although in euclidean space x is horizontal vector
        and y is vertical vector, here we assume x is as the
        same as i which is vertical index of a matrix and
        y is as the same as j which is horizontal index of
        a matrix.
     */
    private int x;
    private int y;
    private TileType type;

    public Tile() {
        this.x = -1;
        this.y = -1;
    }

    public Tile(int x, int y, TileType type) {
        this.x = x;
        this.y = y;
        this.type = type;
    }

    public int getX() {
        return x;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Tile tile = (Tile) o;
        return this.x == tile.x &&
                this.y == tile.y;
    }

    @Override
    public int hashCode() {
        return Objects.hash(x, y);
    }

    @Override
    public String toString() {
        return "(" + this.x + ", " + this.y + ")";
    }

    public void setX(int x) {
        this.x = x;
    }

    public int getY() {
        return y;
    }

    public void setY(int y) {
        this.y = y;
    }

    public TileType getType() {
        return type;
    }

    public void setType(TileType type) {
        this.type = type;
    }
}
