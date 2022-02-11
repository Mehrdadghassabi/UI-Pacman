package com.example.model.dto;

import com.example.model.Tile;

public class Node implements Comparable<Node> {
    private int level;
    private int costUntilHere;
    private Tile currentTile;

    public Node(int level, int costUntilHere, Tile currentTile) {
        this.level = level;
        this.costUntilHere = costUntilHere;
        this.currentTile = currentTile;
    }

    @Override
    public int hashCode() {
        return (20 * this.currentTile.getX()) + this.currentTile.getY();
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || obj.getClass() != this.getClass()) return false;
        Node node = (Node) obj;
        return node.currentTile.getX() == this.currentTile.getX()
                && node.currentTile.getY() == this.currentTile.getY();
    }

    @Override
    public int compareTo(Node node) {
        return Integer.compare(this.costUntilHere, node.costUntilHere);
    }

    public int getLevel() {
        return level;
    }

    public void setLevel(int level) {
        this.level = level;
    }

    public int getCostUntilHere() {
        return costUntilHere;
    }

    public void setCostUntilHere(int costUntilHere) {
        this.costUntilHere = costUntilHere;
    }

    public Tile getCurrentTile() {
        return currentTile;
    }

    public void setCurrentTile(Tile currentTile) {
        this.currentTile = currentTile;
    }
}
