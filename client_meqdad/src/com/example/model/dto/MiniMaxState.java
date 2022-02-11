package com.example.model.dto;

import com.example.BaseAgent;
import com.example.model.Tile;
import com.example.model.TileType;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

@Builder
@Getter
@Setter
public class MiniMaxState {
    private List<BaseAgent.Action> actionsDoneUntilHere;
    private String[][] gridOnThisState;
    private Tile maxPlayerTile;
    private int maxPlayerScore;
    private HashMap<TileType, Integer> maxPlayerCollectedGems;
    private int maxPlayerTrapUsed;
    private Tile minPlayerTile;
    private int minPlayerScore;
    private HashMap<TileType, Integer> minPlayerCollectedGems;
    private int stateLevel;
    private int turnNumber;
    private int diffScores;

    public void customToString() {
        System.out.println("action list: " + actionsDoneUntilHere);
        System.out.println("grid: " + Arrays.deepToString(gridOnThisState));
        System.out.println("max player score: " + maxPlayerScore);
        System.out.println("min player score: " + minPlayerScore);
        System.out.println("level of state: " + stateLevel);
        System.out.println("turn number: " + turnNumber);
        System.out.println("max player expected gems to collect: " + maxPlayerCollectedGems);
        System.out.println("min player expected gems to collect: " + minPlayerCollectedGems);
        System.out.println("diff scores: " + diffScores);
    }
}
