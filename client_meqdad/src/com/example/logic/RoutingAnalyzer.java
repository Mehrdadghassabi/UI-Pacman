package com.example.logic;

import com.example.BaseAgent;
import com.example.Utils;
import com.example.model.*;
import com.example.model.dto.SimpleBFSResponse;

import java.util.*;

public class RoutingAnalyzer {

    private final Utils utils;
    private final OrderingAnalyzer orderingAnalyzer;
    private final static float MAX_TIME_FACTOR = 0.9f;

    public RoutingAnalyzer(Utils utils, OrderingAnalyzer orderingAnalyzer) {
        this.utils = utils;
        this.orderingAnalyzer = orderingAnalyzer;
    }

    public BaseAgent.Action getNextAction(Tile myAgentTile, Tile goalTile) {
        SimpleBFSResponse simpleBfsResponse = utils.bfsSearch(myAgentTile, goalTile, utils.getAgent().getGrid());
        long startTime = System.nanoTime();
        long endTime = System.nanoTime();
        int bfsRunTimes = 1;
        while (simpleBfsResponse == null && (endTime - startTime) < (MAX_TIME_FACTOR * utils.getAgent().getTimeout())) {
            utils.getAgent().getBlockedGems().add(goalTile);
            utils.refreshTileTypeListMap();
            goalTile = orderingAnalyzer.getOptimalGoal(myAgentTile);
            if (goalTile == null)
                break;
            simpleBfsResponse = utils.bfsSearch(myAgentTile, goalTile, utils.getAgent().getGrid());
            endTime = System.nanoTime();
            bfsRunTimes++;
        }

        List<Tile> path;
        if (simpleBfsResponse != null) {
            path = simpleBfsResponse.getPath();
//            System.out.println("turn: " + utils.getAgent().getTurnCount());
//            System.out.println("agent: " + myAgentTile.toString());
//            System.out.println("goal: " + goalTile.toString());
//            System.out.println("path: " + path);
//            System.out.println("cost to gaol: " + simpleBfsResponse.getCost());
//            System.out.println("agent score: " + utils.getAgent().getAgentScores()[0]);
//            System.out.println("time elapsed: " + (endTime - startTime));
//            System.out.println("bfs run times: " + bfsRunTimes);
        } else {
            System.out.println("bfs response is null!");
            return BaseAgent.Action.NoOp;
        }
        if (path.isEmpty()) {
            System.out.println("path is empty!");
            return BaseAgent.Action.NoOp;
        }
        return getNextStep(myAgentTile, path.get(0));
    }

    private BaseAgent.Action getNextStep(Tile source, Tile nextStep) {
        if (source.getX() == nextStep.getX() && source.getY() - nextStep.getY() == -1)
            return BaseAgent.Action.Right;
        else if (source.getX() == nextStep.getX() && source.getY() - nextStep.getY() == 1)
            return BaseAgent.Action.Left;
        else if (source.getY() == nextStep.getY() && source.getX() - nextStep.getX() == -1)
            return BaseAgent.Action.Down;
        else if (source.getY() == nextStep.getY() && source.getX() - nextStep.getX() == 1)
            return BaseAgent.Action.Up;
        else if (source.getType().equals(TileType.TELEPORT) && nextStep.getType().equals(TileType.TELEPORT))
            return BaseAgent.Action.Teleport;
        else throw new IllegalStateException("unknown step!");
    }

}
