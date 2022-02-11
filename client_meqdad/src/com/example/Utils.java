package com.example;

import com.example.model.*;
import com.example.model.dto.*;

import java.util.*;

public class Utils {

    private Agent agent;
    private HashMap<TileType, Integer> myAgentCollectedGemsMap;
    private HashMap<TileType, Integer> opponentCollectedGemsMap;
    private HashMap<TileType, List<Tile>> tileTypeListMap;
    private Tile myAgentTile;
    private Tile opponentTile;
    private int myAgentScore;
    private int opponentScore;
    private String myAgentChar;
    private String opponentChar;
    private String myAgentTrapChar;
    private String opponentTrapChar;
    private final int teleportCost;

    public Utils(Agent agent) {
        this.agent = agent;
        this.initAgentScores();
        this.initAgentsChar();
        this.initAgentTile();

        this.myAgentCollectedGemsMap = new HashMap<>();
        this.opponentCollectedGemsMap = new HashMap<>();
        this.initCollectedGemsMap();

        this.tileTypeListMap = new HashMap<>();
        this.initTileTypeListMap();
        this.updateTileTypeListMap();

        this.teleportCost = this.tileTypeListMap.get(TileType.TELEPORT).size() - 1;
    }

    /*
        this function update all utils properties except
        agent and teleportCost which are constant
     */
    public void refreshTileTypeListMap() {
        this.tileTypeListMap = new HashMap<>();
        this.initTileTypeListMap();
        this.updateTileTypeListMap();
    }

    /*
        this function inits agent tile. we call it
        before initializing other properties because
        we call possibleToGetGem on each gem tile
        which uses agent tile so agent tile should
        not be null
     */
    private void initAgentTile() {
        for (int i = 0; i < this.agent.getGrid().length; i++) {
            for (int j = 0; j < this.agent.getGrid()[0].length; j++) {
                if (this.agent.getGrid()[i][j].contains(String.valueOf(this.myAgentChar))) {
                    TileType type = getTileType(i, j);
                    this.myAgentTile = new Tile(i, j, type);
                }
                if (this.agent.getGrid()[i][j].contains(String.valueOf(this.opponentChar))) {
                    TileType type = getTileType(i, j);
                    this.opponentTile = new Tile(i, j, type);
                }
            }
        }
    }

    private void initTileTypeListMap() {
        for (TileType type : TileType.values()) {
            tileTypeListMap.put(type, new ArrayList<>());
        }
    }

    private void updateTileTypeListMap() {
        for (int i = 0; i < this.agent.getGrid().length; i++) {
            for (int j = 0; j < this.agent.getGrid()[0].length; j++) {
                addTileToTileTypeListMap(i, j);
            }
        }
    }

    private void addTileToTileTypeListMap(int x, int y) {
        TileType type = getTileType(x, y);
        Tile tile = new Tile(x, y, type);
        if (type.equals(TileType.YELLOW_GEM) || type.equals(TileType.GREEN_GEM) ||
                type.equals(TileType.RED_GEM) || type.equals(TileType.BLUE_GEM)) {
            if (possibleToGetGemByMyAgent(this.myAgentTile, tile) && !this.agent.getBlockedGems().contains(tile))
                this.tileTypeListMap.get(type).add(tile);
        } else
                this.tileTypeListMap.get(type).add(tile);
    }

    private void initCollectedGemsMap() {
        int index = 0;
        for (TileType type : TileType.values()) {
            if (type.equals(TileType.YELLOW_GEM) || type.equals(TileType.GREEN_GEM) ||
                    type.equals(TileType.RED_GEM) || type.equals(TileType.BLUE_GEM)) {
                if (this.agent.getAgentCount() == 2) {
                    if (this.agent.getCharacter() == 'A') {
                        this.myAgentCollectedGemsMap.put(type, this.agent.getAgentGems()[0][index]);
                        this.opponentCollectedGemsMap.put(type, this.agent.getAgentGems()[1][index]);
                    } else {
                        this.myAgentCollectedGemsMap.put(type, this.agent.getAgentGems()[1][index]);
                        this.opponentCollectedGemsMap.put(type, this.agent.getAgentGems()[0][index]);
                    }
                } else {
                    this.myAgentCollectedGemsMap.put(type, this.agent.getAgentGems()[0][index]);
                }
                index++;
            }
        }
    }

    private void initAgentScores() {
        if (this.agent.getAgentCount() == 2) {
            if (this.agent.getCharacter() == 'A') {
                this.myAgentScore = this.agent.getAgentScores()[0];
                this.opponentScore = this.agent.getAgentScores()[1];
            } else {
                this.myAgentScore = this.agent.getAgentScores()[1];
                this.opponentScore = this.agent.getAgentScores()[0];
            }
        } else {
            this.myAgentScore = this.agent.getAgentScores()[0];
        }
    }
    
    private void initAgentsChar() {
        if (this.agent.getCharacter() == 'A') {
            this.myAgentChar = "A";
            this.opponentChar = "B";
            this.myAgentTrapChar = "a";
            this.opponentTrapChar = "b";
        } else {
            this.myAgentChar = "B";
            this.opponentChar = "A";
            this.myAgentTrapChar = "b";
            this.opponentTrapChar = "a";
        }
    }

    /*
        a branch and bound algorithm to get optimal order of picking gems.
        update this algorithm to consider gem rewards and remained turns.
     */
    public ComplexBFSNode getOptimalOrder(Tile source, List<Tile> destinationGems, String[][] grid,
                                      int turnNumber, int agentScore) {
        if (destinationGems == null)
            return null;

        PriorityQueue<ComplexBFSNode> queue = new PriorityQueue<>();
        List<Tile> pickedGems = new LinkedList<>();
        ComplexBFSNode startNode = new ComplexBFSNode(0, 0, 0, agentScore, source,
                pickedGems, destinationGems);
        startNode.setBound(bound(startNode));
        queue.add(startNode);
        int remainedTurns = this.agent.getMaxTurnCount() - turnNumber;
        int maximumReward = Integer.MIN_VALUE;
        ComplexBFSNode bestNode = new ComplexBFSNode(0, 0, 0, agentScore, source,
                pickedGems, destinationGems);

        while (!queue.isEmpty()) {
            ComplexBFSNode current = queue.remove();
            if (current.getBound() > maximumReward) {
                for (Tile childTile : current.getRemainedGems()) {
                    int costFromCurrentToChild = costFromParentToChild(current.getCurrentTile(), childTile, grid);
                    int costUntilHere = current.getCostUntilHere() + costFromCurrentToChild;
                    int rewardUntilHere = current.getRewardUntilHere() + this.getGemRewardByType(childTile.getType());

                    // here we do not need to consider collectedGemsMap,
                    // we have considered it when we calculated destinationGems.
                    if (!this.possibleToGetGemByExactDistanceWithoutCollectedGemsMap(childTile, current.getAgentScore(),
                            costFromCurrentToChild))
                        continue;

                    int childAgentScore = current.getAgentScore() - costFromCurrentToChild +
                            this.getGemRewardByType(childTile.getType());

                    List<Tile> childPickedGems = new LinkedList<>(current.getPickedGems());
                    childPickedGems.add(childTile);
                    List<Tile> remainedChildGems = new LinkedList<>(current.getRemainedGems());
                    remainedChildGems.remove(childTile);
                    ComplexBFSNode childNode = new ComplexBFSNode(current.getLevel() + 1,
                            costUntilHere,
                            rewardUntilHere,
                            childAgentScore,
                            childTile,
                            childPickedGems,
                            remainedChildGems);
                    childNode.setBound(bound(childNode));

                    if (costUntilHere > remainedTurns)
                        continue;

                    if (childNode.getRewardUntilHere() - childNode.getCostUntilHere() > maximumReward) {
                        maximumReward = childNode.getRewardUntilHere() - childNode.getCostUntilHere();
                        bestNode = childNode;
                    }

                    if (bound(childNode) > maximumReward)
                        queue.add(childNode);
                }
            }
        }

        return bestNode;
    }

    // dependency is ok
    private int costFromParentToChild(Tile parent, Tile child, String[][] grid) {
        SimpleBFSResponse simpleBfsResponse = bfsSearch(parent, child, grid);
        if (simpleBfsResponse != null) return simpleBfsResponse.getCost();
        else throw new NullPointerException("bfs response is null!");
    }

    // dependency is ok
    private int bound(ComplexBFSNode node) {
        List<Tile> remainedGems = new LinkedList<>(node.getRemainedGems());
        remainedGems.add(node.getCurrentTile());
        int costUntilHere = node.getCostUntilHere();
        int rewardUntilHere = node.getRewardUntilHere();
        int minRemainedCost = 0;
        int maxRemainedReward = 0;
        for (int i = 0; i < remainedGems.size(); i++) {
            int min = Integer.MAX_VALUE;
            if (!remainedGems.get(i).equals(node.getCurrentTile())) {
                maxRemainedReward += this.getGemRewardByType(remainedGems.get(i).getType());
            }
            for (int j = 0; j <remainedGems.size(); j++) {
                if (i == j)
                    continue;
                Tile iTile = remainedGems.get(i);
                Tile jTile = remainedGems.get(j);
                int cost = Math.abs(iTile.getX() - jTile.getX()) + Math.abs(iTile.getY() - jTile.getY());
                if (cost < min)
                    min = cost;
            }
            minRemainedCost += min;
        }

        return (rewardUntilHere + maxRemainedReward) - (costUntilHere);
    }

    // dependency is ok
    public List<Tile> BFSToGetTopNearestReachableGems(Tile start, int maxGemsToExplore, String[][] grid,
                                                      HashMap<TileType, Integer> collectedGemsMap, int agentScore) {
        PriorityQueue<SimpleBFSNode> frontier = new PriorityQueue<>();
        Set<Tile> exploredTiles = new HashSet<>();
        List<Tile> exploredGems = new LinkedList<>();
        HashMap<TileType, Integer> collectedGemsMapCopy = this.hashMapDeepCopy(collectedGemsMap);
        List<Tile> cameFrom = new LinkedList<>();
        SimpleBFSNode startNode = new SimpleBFSNode(0, 0, start, cameFrom);

        frontier.add(startNode);
        exploredTiles.add(start);

        while (!frontier.isEmpty()) {
            SimpleBFSNode current = frontier.remove();

            if (exploredGems.size() == maxGemsToExplore)
                return exploredGems;

            if (this.isTileTypeOfGem(current.getCurrentTile()) &&
                    possibleToGetGemByExactDistance(current.getCurrentTile(), collectedGemsMapCopy,
                            agentScore, current.getCostUntilHere())) {
                TileType gemType = current.getCurrentTile().getType();
                exploredGems.add(current.getCurrentTile());
                collectedGemsMapCopy.replace(gemType, collectedGemsMapCopy.get(gemType) + 1);
                agentScore += this.getGemRewardByType(gemType);
            }
            exploredTiles.add(current.getCurrentTile());

            for (SimpleBFSNode successor : getNeighbors(current, grid)) {

                if (!exploredTiles.contains(successor.getCurrentTile()) && !(frontier.contains(successor))) {
                    frontier.add(successor);
                } else if (frontier.contains(successor)) {
                    frontier.forEach(node -> {
                        if (node.equals(successor) && node.getCostUntilHere() > successor.getCostUntilHere()) {
                            updateFrontier(successor, node);
                        }
                    });
                }
            }
        }

        return exploredGems;
    }

    // dependency is ok
    public SimpleBFSResponse bfsSearch(Tile source, Tile destination, String[][] grid) {
        PriorityQueue<SimpleBFSNode> frontier = new PriorityQueue<>();
        Set<Tile> exploredSet = new HashSet<>();

        List<Tile> cameFrom = new LinkedList<>();
        frontier.add(new SimpleBFSNode(0, 0, source, cameFrom));
        exploredSet.add(source);

        while (!frontier.isEmpty()) {
            SimpleBFSNode current = frontier.remove();

            if (current.getCurrentTile().equals(destination))
                return new SimpleBFSResponse(
                        current.getCameFrom(),
                        current.getCostUntilHere(),
                        destination,
                        source,
                        current.getLevel());

            exploredSet.add(current.getCurrentTile());
            for (SimpleBFSNode successor : getNeighbors(current, grid)) {

                if (!exploredSet.contains(successor.getCurrentTile()) && !(frontier.contains(successor))) {
                    frontier.add(successor);
                } else if (frontier.contains(successor)) {
                    frontier.forEach(node -> {
                        if (node.equals(successor) && node.getCostUntilHere() > successor.getCostUntilHere()) {
                            updateFrontier(successor, node);
                        }
                    });
                }
            }
        }
        return null;
    }

    private void updateFrontier(SimpleBFSNode successor, SimpleBFSNode node) {
        node.setLevel(successor.getLevel());
        node.setCameFrom(successor.getCameFrom());
        node.setCostUntilHere(successor.getCostUntilHere());
    }

    // dependency is ok
    private List<SimpleBFSNode> getNeighbors(SimpleBFSNode current, String[][] grid) {
        List<SimpleBFSNode> neighbors = new LinkedList<>();
        if (current.getCurrentTile().getType().equals(TileType.TELEPORT)) {
            neighbors.addAll(getTeleportNeighbors(current));
        }
        neighbors.addAll(getPhysicalNeighbors(current, grid));
        return neighbors;
    }

    // dependency is ok
    private List<SimpleBFSNode> getPhysicalNeighbors(SimpleBFSNode currentNode, String[][] grid) {
        List<SimpleBFSNode> neighbors = new LinkedList<>();
        Tile currentTile = currentNode.getCurrentTile();
        if (currentTile.getY() - 1 >= 0) {
            addNeighborTile(neighbors, currentTile.getX(), currentTile.getY() - 1, currentNode, grid);
        }
        if (currentTile.getY() + 1 < grid[0].length) {
            addNeighborTile(neighbors, currentTile.getX(), currentTile.getY() + 1, currentNode, grid);
        }
        if (currentTile.getX() - 1 >= 0) {
            addNeighborTile(neighbors, currentTile.getX() - 1, currentTile.getY(), currentNode, grid);
        }
        if (currentTile.getX() + 1 < grid.length) {
            addNeighborTile(neighbors, currentTile.getX() + 1, currentTile.getY(), currentNode, grid);
        }
        return neighbors;
    }

    // dependency is ok
    private void addNeighborTile(List<SimpleBFSNode> neighbors, int x, int y, SimpleBFSNode currentNode, String[][] grid) {
        Tile neighbor;
        TileType type = this.getTileType(x, y, grid);
        if (type != TileType.WALL) {
            neighbor = new Tile(x, y, type);
            List<Tile> cameFrom = new LinkedList<>(currentNode.getCameFrom());
            cameFrom.add(neighbor);
            neighbors.add(new SimpleBFSNode(currentNode.getLevel() + 1, currentNode.getCostUntilHere() + 1,
                    neighbor, cameFrom));
        }
    }

    // dependency is ok
    private List<SimpleBFSNode> getTeleportNeighbors(SimpleBFSNode currentTeleportNode) {
        List<Tile> neighborTeleportTiles = new LinkedList<>(this.getTileTypeListMap().get(TileType.TELEPORT));
        neighborTeleportTiles.remove(currentTeleportNode.getCurrentTile());
        List<SimpleBFSNode> neighborTeleportNodes = new LinkedList<>();
        for (Tile teleportTile : neighborTeleportTiles) {
            List<Tile> cameFrom = new LinkedList<>(currentTeleportNode.getCameFrom());
            cameFrom.add(teleportTile);
            neighborTeleportNodes.add(new SimpleBFSNode(currentTeleportNode.getLevel() + 1,
                    currentTeleportNode.getCostUntilHere() + this.getTeleportCost(),
                    teleportTile, cameFrom));
        }
        return neighborTeleportNodes;
    }

    public TileType getTileType(int x, int y) {
        return getTileType(x, y, this.agent.getGrid());
    }

    public TileType getTileType(int x, int y, String[][] grid) {
        if (grid[x][y].contains(TileType.EMPTY.getValue()))
            return TileType.EMPTY;
        else if (grid[x][y].contains(TileType.TELEPORT.getValue()))
            return TileType.TELEPORT;
        else if (grid[x][y].contains(TileType.YELLOW_GEM.getValue()))
            return TileType.YELLOW_GEM;
        else if (grid[x][y].contains(TileType.GREEN_GEM.getValue()))
            return TileType.GREEN_GEM;
        else if (grid[x][y].contains(TileType.RED_GEM.getValue()))
            return TileType.RED_GEM;
        else if (grid[x][y].contains(TileType.BLUE_GEM.getValue()))
            return TileType.BLUE_GEM;
        else if (grid[x][y].contains(TileType.WALL.getValue()))
            return TileType.WALL;
        else throw new IllegalStateException("Tile isn't valid!");
    }

    public boolean possibleToGetGemByMyAgent(Tile agentTile, Tile goalTile) {
        int myAgentScore;
        if (this.agent.getCharacter() == 'A')
            myAgentScore = this.agent.getAgentScores()[0];
        else
            myAgentScore = this.agent.getAgentScores()[1];
        return possibleToGetGemByHeuristicDistance(agentTile, goalTile, this.myAgentCollectedGemsMap, myAgentScore);
    }

    public boolean possibleToGetGemByOpponent(Tile agentTile, Tile goalTile) {
        int opponentScore;
        if (this.agent.getCharacter() == 'A')
            opponentScore = this.agent.getAgentScores()[1];
        else
            opponentScore = this.agent.getAgentScores()[0];
        return possibleToGetGemByHeuristicDistance(agentTile, goalTile, this.opponentCollectedGemsMap, opponentScore);
    }

    public boolean possibleToGetGemByHeuristicDistance(Tile agentTile, Tile gemTile,
                                                       Map<TileType, Integer> collectedGemsMap, int agentScore) {
        int minCostToGoal = heuristic(agentTile, gemTile);
        int requiredScore = this.getGemRequiredScoreByType(gemTile.getType());
        if ((agentScore - minCostToGoal) >= requiredScore) {
            if (gemTile.getType().equals(TileType.YELLOW_GEM)) {
                return collectedGemsMap.get(TileType.YELLOW_GEM) < YellowGemTile.MAXIMUM_ACHIEVABLE_TIMES.getValue();
            } else if (gemTile.getType().equals(TileType.GREEN_GEM)) {
                return collectedGemsMap.get(TileType.GREEN_GEM) < GreenGemTile.MAXIMUM_ACHIEVABLE_TIMES.getValue();
            } else if (gemTile.getType().equals(TileType.RED_GEM)) {
                return collectedGemsMap.get(TileType.RED_GEM) < RedGemTile.MAXIMUM_ACHIEVABLE_TIMES.getValue();
            } else if (gemTile.getType().equals(TileType.BLUE_GEM)) {
                return collectedGemsMap.get(TileType.BLUE_GEM) < BlueGemTile.MAXIMUM_ACHIEVABLE_TIMES.getValue();
            } else throw new IllegalStateException("goal should be gem!");
        }
        return false;
    }

    public boolean possibleToGetGemByExactDistance(Tile gemTile, Map<TileType, Integer> collectedGemsMap,
                                                   int agentScore, int distance) {
        int requiredScore = this.getGemRequiredScoreByType(gemTile.getType());
        if ((agentScore - distance) >= requiredScore) {
            if (gemTile.getType().equals(TileType.YELLOW_GEM)) {
                return collectedGemsMap.get(TileType.YELLOW_GEM) < YellowGemTile.MAXIMUM_ACHIEVABLE_TIMES.getValue();
            } else if (gemTile.getType().equals(TileType.GREEN_GEM)) {
                return collectedGemsMap.get(TileType.GREEN_GEM) < GreenGemTile.MAXIMUM_ACHIEVABLE_TIMES.getValue();
            } else if (gemTile.getType().equals(TileType.RED_GEM)) {
                return collectedGemsMap.get(TileType.RED_GEM) < RedGemTile.MAXIMUM_ACHIEVABLE_TIMES.getValue();
            } else if (gemTile.getType().equals(TileType.BLUE_GEM)) {
                return collectedGemsMap.get(TileType.BLUE_GEM) < BlueGemTile.MAXIMUM_ACHIEVABLE_TIMES.getValue();
            } else throw new IllegalStateException("goal should be gem!");
        }
        return false;
    }

    public boolean possibleToGetGemByExactDistanceWithoutCollectedGemsMap(Tile gemTile,
                                                   int agentScore, int distance) {
        int requiredScore = this.getGemRequiredScoreByType(gemTile.getType());
        return (agentScore - distance) >= requiredScore;
    }

    public int heuristic(Tile start, Tile end) {
        return Math.abs(start.getX() - end.getX()) + Math.abs(start.getY() - end.getY());
    }

    public boolean isRightActionValid(Tile tile, String[][] grid) {
        int currentTileX = tile.getX();
        int currentTileY = tile.getY();
        int rightTileY = currentTileY + 1;
        // not getting out of grid
        if (rightTileY < grid[0].length) {
            TileType type = this.getTileType(currentTileX, rightTileY, grid);
            // not encountering wall
            return type != TileType.WALL;
        }
        return false;
    }

    public boolean isUpActionValid(Tile tile, String[][] grid) {
        int currentTileX = tile.getX();
        int currentTileY = tile.getY();
        int upTileX = currentTileX - 1;
        // not getting out of grid
        if (upTileX >= 0) {
            TileType type = this.getTileType(upTileX, currentTileY, grid);
            // not encountering wall
            return type != TileType.WALL;
        }
        return false;
    }

    public boolean isLeftActionValid(Tile tile, String[][] grid) {
        int currentTileX = tile.getX();
        int currentTileY = tile.getY();
        int leftTileY = currentTileY - 1;
        // not getting out of grid
        if (leftTileY >= 0) {
            TileType type = this.getTileType(currentTileX, leftTileY, grid);
            // not encountering wall
            return type != TileType.WALL;
        }
        return false;
    }

    public boolean isDownActionValid(Tile tile, String[][] grid) {
        int currentTileX = tile.getX();
        int currentTileY = tile.getY();
        int downTileX = currentTileX + 1;
        // not getting out of grid
        if (downTileX < grid.length) {
            TileType type = this.getTileType(downTileX, currentTileY, grid);
            // not encountering wall
            return type != TileType.WALL;
        }
        return false;
    }

    public boolean isTeleportActionValid(Tile tile, String[][] grid) {
        return getTileType(tile.getX(), tile.getY(), grid) == TileType.TELEPORT;
    }

    public boolean isTrapActionValid(Tile tile, MiniMaxState miniMaxState) {
        TileType type = getTileType(tile.getX(), tile.getY(), miniMaxState.getGridOnThisState());
        // not being in teleport tile and having remained traps
        if (type != TileType.TELEPORT && this.agent.getTurnCount() - miniMaxState.getMaxPlayerTrapUsed() > 0) {
            int requiredScoreForTrap = 35 * (miniMaxState.getMaxPlayerTrapUsed() + 1);
            // having required score for trap
            return requiredScoreForTrap >= miniMaxState.getMaxPlayerScore();
        }
        return false;
    }

    public boolean isTileTypeOfGem(Tile tile) {
        return tile.getType().equals(TileType.YELLOW_GEM) || tile.getType().equals(TileType.GREEN_GEM) ||
                tile.getType().equals(TileType.RED_GEM) || tile.getType().equals(TileType.BLUE_GEM);
    }

    public boolean isTileTypeOfGemAccordingToStringValue(String tileString) {
        return tileString.contains(TileType.YELLOW_GEM.getValue()) ||
                tileString.contains(TileType.GREEN_GEM.getValue()) ||
                tileString.contains(TileType.RED_GEM.getValue()) ||
                tileString.contains(TileType.BLUE_GEM.getValue());
    }

    public TileType getGemTypeByTileStringValue(String tileString) {
        if (tileString.charAt(0) == '1')
            return TileType.YELLOW_GEM;
        else if (tileString.charAt(0) == '2')
            return TileType.GREEN_GEM;
        else if (tileString.charAt(0) == '3')
            return TileType.RED_GEM;
        else if (tileString.charAt(0) == '4')
            return TileType.BLUE_GEM;
        else
            throw new IllegalStateException("string value is not related to gem!");
    }

    public int getGemRewardByType(TileType type) {
        if (type.equals(TileType.YELLOW_GEM))
            return YellowGemTile.REWARD.getValue();
        else if (type.equals(TileType.GREEN_GEM))
            return GreenGemTile.REWARD.getValue();
        else if (type.equals(TileType.RED_GEM))
            return RedGemTile.REWARD.getValue();
        else if (type.equals(TileType.BLUE_GEM))
            return BlueGemTile.REWARD.getValue();
        else
            throw new IllegalStateException("tile should be gem to have reward!");
    }

    public int getGemRequiredScoreByType(TileType type) {
        if (type.equals(TileType.YELLOW_GEM))
            return YellowGemTile.REQUIRED_SCORE.getValue();
        else if (type.equals(TileType.GREEN_GEM))
            return GreenGemTile.REQUIRED_SCORE.getValue();
        else if (type.equals(TileType.RED_GEM))
            return RedGemTile.REQUIRED_SCORE.getValue();
        else if (type.equals(TileType.BLUE_GEM))
            return BlueGemTile.REQUIRED_SCORE.getValue();
        else
            throw new IllegalStateException("tile should be gem to have reward!");
    }

    public int getGemAchievableTimesByType(TileType type) {
        if (type.equals(TileType.YELLOW_GEM))
            return YellowGemTile.MAXIMUM_ACHIEVABLE_TIMES.getValue();
        else if (type.equals(TileType.GREEN_GEM))
            return GreenGemTile.MAXIMUM_ACHIEVABLE_TIMES.getValue();
        else if (type.equals(TileType.RED_GEM))
            return RedGemTile.MAXIMUM_ACHIEVABLE_TIMES.getValue();
        else if (type.equals(TileType.BLUE_GEM))
            return BlueGemTile.MAXIMUM_ACHIEVABLE_TIMES.getValue();
        else
            throw new IllegalStateException("tile should be gem to have reward!");
    }

    public String getOpponentCharBySource(String source) {
        if (source.contains("A"))
            return "B";
        else
            return "A";
    }

    public String getOpponentTrapCharBySource(String source) {
        if (source.contains("A"))
            return "b";
        else
            return "a";
    }

    public String getMyAgentCharBySource(String source) {
        if (source.contains("A"))
            return "A";
        else
            return "B";
    }

    public PlayerType getPlayerTypeByTile(String[][] grid, Tile player) {
        if (grid[player.getX()][player.getY()].contains(this.myAgentChar))
            return PlayerType.MAX_PLAYER;
        else if (grid[player.getX()][player.getY()].contains(this.opponentChar))
            return PlayerType.MIN_PLAYER;
        else
            throw new IllegalStateException("grid or player tile is not valid!");
    }

    public Tile copyTile(Tile tile) {
        return new Tile(tile.getX(), tile.getY(), tile.getType());
    }

    public HashMap<TileType, Integer> hashMapDeepCopy(HashMap<TileType, Integer> originalMap) {
        HashMap<TileType, Integer> newMap = new HashMap<>();
        Set<Map.Entry<TileType, Integer>> entries = originalMap.entrySet();
        for (Map.Entry<TileType, Integer> entry : entries) {
            newMap.put(entry.getKey(), entry.getValue());
        }
        return newMap;
    }

    public HashMap<QStateAction, Float> qTableDeepCopy(HashMap<QStateAction, Float> originalMap) {
        HashMap<QStateAction, Float> newMap = new HashMap<>();
        Set<Map.Entry<QStateAction, Float>> entries = originalMap.entrySet();
        for (Map.Entry<QStateAction, Float> entry : entries) {
            newMap.put(entry.getKey(), entry.getValue());
        }
        return newMap;
    }

    public String[][] gridDeepCopy(String[][] oldGrid) {
        String[][] newGrid = new String[oldGrid.length][oldGrid[0].length];
        for (int i = 0; i < newGrid.length; i++) {
            System.arraycopy(oldGrid[i], 0, newGrid[i], 0, newGrid[0].length);
        }
        return newGrid;
    }

    @SafeVarargs
    private List<Tile> concatenateLists(List<Tile>... lists) {
        List<Tile> result = new ArrayList<>();
        for (List<Tile> list : lists)
            result.addAll(list);
        return result;
    }

    public Agent getAgent() {
        return agent;
    }

    public void setAgent(Agent agent) {
        this.agent = agent;
    }

    public HashMap<TileType, List<Tile>> getTileTypeListMap() {
        return tileTypeListMap;
    }

    public void setTileTypeListMap(HashMap<TileType, List<Tile>> tileTypeListMap) {
        this.tileTypeListMap = tileTypeListMap;
    }

    public Tile getMyAgentTile() {
        return myAgentTile;
    }

    public void setMyAgentTile(Tile myAgentTile) {
        this.myAgentTile = myAgentTile;
    }

    public int getTeleportCost() {
        return teleportCost;
    }

    public HashMap<TileType, Integer> getMyAgentCollectedGemsMap() {
        return myAgentCollectedGemsMap;
    }

    public void setMyAgentCollectedGemsMap(HashMap<TileType, Integer> myAgentCollectedGemsMap) {
        this.myAgentCollectedGemsMap = myAgentCollectedGemsMap;
    }

    public HashMap<TileType, Integer> getOpponentCollectedGemsMap() {
        return opponentCollectedGemsMap;
    }

    public void setOpponentCollectedGemsMap(HashMap<TileType, Integer> opponentCollectedGemsMap) {
        this.opponentCollectedGemsMap = opponentCollectedGemsMap;
    }

    public int getMyAgentScore() {
        return myAgentScore;
    }

    public void setMyAgentScore(int myAgentScore) {
        this.myAgentScore = myAgentScore;
    }

    public int getOpponentScore() {
        return opponentScore;
    }

    public void setOpponentScore(int opponentScore) {
        this.opponentScore = opponentScore;
    }

    public String getMyAgentChar() {
        return myAgentChar;
    }

    public void setMyAgentChar(String myAgentChar) {
        this.myAgentChar = myAgentChar;
    }

    public String getOpponentChar() {
        return opponentChar;
    }

    public void setOpponentChar(String opponentChar) {
        this.opponentChar = opponentChar;
    }

    public String getMyAgentTrapChar() {
        return myAgentTrapChar;
    }

    public void setMyAgentTrapChar(String myAgentTrapChar) {
        this.myAgentTrapChar = myAgentTrapChar;
    }

    public String getOpponentTrapChar() {
        return opponentTrapChar;
    }

    public void setOpponentTrapChar(String opponentTrapChar) {
        this.opponentTrapChar = opponentTrapChar;
    }

    public Tile getOpponentTile() {
        return opponentTile;
    }

    public void setOpponentTile(Tile opponentTile) {
        this.opponentTile = opponentTile;
    }
}
