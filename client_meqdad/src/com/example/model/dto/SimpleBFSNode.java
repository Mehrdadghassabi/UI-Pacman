package com.example.model.dto;

import com.example.model.Tile;

import java.util.List;

public class SimpleBFSNode extends Node {
    private List<Tile> cameFrom;

    public SimpleBFSNode(int level, int costUntilHere, Tile currentTile, List<Tile> cameFrom) {
        super(level, costUntilHere, currentTile);
        this.cameFrom = cameFrom;
    }

    public List<Tile> getCameFrom() {
        return cameFrom;
    }

    public void setCameFrom(List<Tile> cameFrom) {
        this.cameFrom = cameFrom;
    }
}
