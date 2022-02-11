package com.example.model.dto;

import com.example.model.Tile;

import java.util.List;

public class ComplexBFSNode extends Node {
    private List<Tile> pickedGems;
    private List<Tile> remainedGems;
    private int rewardUntilHere;
    private int agentScore;
    private int bound;

    public ComplexBFSNode(int level, int costUntilHere, int rewardUntilHere, int agentScore, Tile currentTile,
                          List<Tile> pickedGems, List<Tile> remainedGems) {
        super(level, costUntilHere, currentTile);
        this.pickedGems = pickedGems;
        this.remainedGems = remainedGems;
        this.rewardUntilHere = rewardUntilHere;
        this.agentScore = agentScore;
    }

    public List<Tile> getPickedGems() {
        return pickedGems;
    }

    public void setPickedGems(List<Tile> pickedGems) {
        this.pickedGems = pickedGems;
    }

    public List<Tile> getRemainedGems() {
        return remainedGems;
    }

    public void setRemainedGems(List<Tile> remainedGems) {
        this.remainedGems = remainedGems;
    }

    public int getBound() {
        return bound;
    }

    public void setBound(int bound) {
        this.bound = bound;
    }

    public int getRewardUntilHere() {
        return rewardUntilHere;
    }

    public void setRewardUntilHere(int rewardUntilHere) {
        this.rewardUntilHere = rewardUntilHere;
    }

    public int getAgentScore() {
        return agentScore;
    }

    public void setAgentScore(int agentScore) {
        this.agentScore = agentScore;
    }
}
