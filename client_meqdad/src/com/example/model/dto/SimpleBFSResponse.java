package com.example.model.dto;

import com.example.model.Tile;

import java.util.List;

public class SimpleBFSResponse {
    private List<Tile> path;
    private int cost;
    private Tile destination;
    private Tile source;
    private int goalLevel;

    public SimpleBFSResponse(List<Tile> path, int cost, Tile destination, Tile source, int goalLevel) {
        this.path = path;
        this.cost = cost;
        this.destination = destination;
        this.source = source;
        this.goalLevel = goalLevel;
    }

    public List<Tile> getPath() {
        return path;
    }

    public void setPath(List<Tile> path) {
        this.path = path;
    }

    public int getCost() {
        return cost;
    }

    public void setCost(int cost) {
        this.cost = cost;
    }

    public Tile getDestination() {
        return destination;
    }

    public void setDestination(Tile destination) {
        this.destination = destination;
    }

    public Tile getSource() {
        return source;
    }

    public void setSource(Tile source) {
        this.source = source;
    }

    public int getGoalLevel() {
        return goalLevel;
    }

    public void setGoalLevel(int goalLevel) {
        this.goalLevel = goalLevel;
    }
}
