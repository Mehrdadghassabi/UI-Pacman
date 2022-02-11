package com.example.logic;

import com.example.Utils;
import com.example.model.*;
import com.example.model.dto.ComplexBFSNode;

import java.util.*;

public class OrderingAnalyzer {

    private final Utils utils;
    private final static int MAX_GEMS_TO_BE_EXPLORED_THRESHOLD_FOR_ORDERING_ANALYZER = 4;
    private final static int MAX_GEMS_TO_BE_EXPLORED_THRESHOLD_FOR_ADVERSARIAL_ANALYZER = 3;

    public OrderingAnalyzer(Utils utils) {
        this.utils = utils;
    }

    public Tile getOptimalGoal(Tile myAgentTile) {
        List<Tile> gemsToBeOrdered = utils.BFSToGetTopNearestReachableGems(myAgentTile,
                MAX_GEMS_TO_BE_EXPLORED_THRESHOLD_FOR_ORDERING_ANALYZER,
                utils.getAgent().getGrid(), this.utils.getMyAgentCollectedGemsMap(), this.utils.getMyAgentScore());
        ComplexBFSNode complexBFSNode = utils.getOptimalOrder(myAgentTile, gemsToBeOrdered, utils.getAgent().getGrid(),
                utils.getAgent().getTurnCount(), this.utils.getMyAgentScore());
//        System.out.println("optimal order: " + complexBFSNode.getPickedGems());
        if (!complexBFSNode.getPickedGems().isEmpty())
            return complexBFSNode.getPickedGems().get(0);
        else return null;
    }

    public ComplexBFSNode getOptimalOrderForAdversarialAnalyzer(Tile player, String[][] grid,
                                                                HashMap<TileType, Integer> collectedGemsMap, int agentScore,
                                                                int turnNumber) {
        List<Tile> gemsToBeOrdered = utils.BFSToGetTopNearestReachableGems(player,
                MAX_GEMS_TO_BE_EXPLORED_THRESHOLD_FOR_ADVERSARIAL_ANALYZER,
                grid, collectedGemsMap, agentScore);
        return utils.getOptimalOrder(player, gemsToBeOrdered, grid,
                turnNumber, agentScore);
    }

}
