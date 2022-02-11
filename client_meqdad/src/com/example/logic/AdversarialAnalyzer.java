package com.example.logic;

import com.example.BaseAgent;
import com.example.Utils;
import com.example.model.PlayerType;
import com.example.model.Tile;
import com.example.model.TileType;
import com.example.model.dto.ComplexBFSNode;
import com.example.model.dto.MiniMaxState;

import java.util.*;

public class AdversarialAnalyzer {

    private final Utils utils;
    private final OrderingAnalyzer orderingAnalyzer;
    private final static int CUT_OFF_THRESHOLD = 4;
    private final static int HIT_DAMAGE = 20;
    private final static int TRAP_DAMAGE = 40;
    private final static int TURN_COST = 1;

    public AdversarialAnalyzer(Utils utils, OrderingAnalyzer orderingAnalyzer) {
        this.utils = utils;
        this.orderingAnalyzer = orderingAnalyzer;
    }

    public BaseAgent.Action getNextAction() {
        MiniMaxState root = MiniMaxState.builder()
                .actionsDoneUntilHere(new LinkedList<>())
                .gridOnThisState(this.utils.gridDeepCopy(this.utils.getAgent().getGrid()))
                .maxPlayerTile(this.utils.copyTile(this.utils.getMyAgentTile()))
                .maxPlayerScore(this.utils.getMyAgentScore())
                .maxPlayerCollectedGems(this.utils.hashMapDeepCopy(this.utils.getMyAgentCollectedGemsMap()))
                .maxPlayerTrapUsed(this.utils.getAgent().getTrapUsed())
                .minPlayerTile(this.utils.copyTile(this.utils.getOpponentTile()))
                .minPlayerScore(this.utils.getOpponentScore())
                .minPlayerCollectedGems(this.utils.hashMapDeepCopy(this.utils.getOpponentCollectedGemsMap()))
                .stateLevel(0)
                .turnNumber(this.utils.getAgent().getTurnCount())
                .build();
        MiniMaxState optimalState = minimax(root);
        if (optimalState != null) {
//            optimalState.customToString();
            if (optimalState.getActionsDoneUntilHere() != null && !optimalState.getActionsDoneUntilHere().isEmpty()) {
                return optimalState.getActionsDoneUntilHere().get(0);
            }
        }
        return null;
    }

    private MiniMaxState minimax(MiniMaxState miniMaxState) {
        return maxValue(miniMaxState, Integer.MIN_VALUE, Integer.MAX_VALUE);
    }

    private MiniMaxState maxValue(MiniMaxState miniMaxState, int alpha, int beta) {
        if (cutOffTest(miniMaxState)) return eval(miniMaxState);
        // this way, state v will never become maximum state in getMaxState function
        MiniMaxState v = MiniMaxState.builder()
                .maxPlayerScore(-1000000)
                .minPlayerScore(1000000)
                .diffScores(-1000000)
                .build();
        Tile maxPlayerTile = miniMaxState.getMaxPlayerTile();
        for (BaseAgent.Action action : getPossibleActions(miniMaxState, maxPlayerTile)) {
            if (action.equals(BaseAgent.Action.Teleport)) {
                for (MiniMaxState teleportResult : resultsForTeleportAction(miniMaxState, action, PlayerType.MAX_PLAYER)) {
                    v = getMaxState(v, minValue(teleportResult, alpha, beta));
                    // watch out
                    if (v.getDiffScores() >= beta)
                        return v;
                    alpha = Math.max(alpha, v.getDiffScores());
                }
            } else {
                v = getMaxState(v, minValue(resultForActionsExceptTeleport(miniMaxState, action, PlayerType.MAX_PLAYER),
                        alpha, beta));
                // watch out
                if (v.getDiffScores() >= beta)
                    return v;
                alpha = Math.max(alpha, v.getDiffScores());
            }
        }
        return v;
    }

    private MiniMaxState minValue(MiniMaxState miniMaxState, int alpha, int beta) {
        if (cutOffTest(miniMaxState)) return eval(miniMaxState);
        // this way, state v will never become maximum state in getMaxState function
        MiniMaxState v = MiniMaxState.builder()
                .maxPlayerScore(1000000)
                .minPlayerScore(-1000000)
                .diffScores(1000000)
                .build();
        Tile minPlayerTile = miniMaxState.getMinPlayerTile();
        for (BaseAgent.Action action : getPossibleActions(miniMaxState, minPlayerTile)) {
            if (action.equals(BaseAgent.Action.Teleport)) {
                for (MiniMaxState teleportResult : resultsForTeleportAction(miniMaxState, action, PlayerType.MIN_PLAYER)) {
                    v = getMinState(v, maxValue(teleportResult, alpha, beta));
                    // watch out
                    if (v.getDiffScores() <= alpha)
                        return v;
                    beta = Math.min(beta, v.getDiffScores());
                }
            } else {
                v = getMinState(v, maxValue(resultForActionsExceptTeleport(miniMaxState, action, PlayerType.MIN_PLAYER),
                        alpha, beta));
                // watch out
                if (v.getDiffScores() <= alpha)
                    return v;
                beta = Math.min(beta, v.getDiffScores());
            }
        }
        return v;
    }

    /*
        needs to be rechecked.
     */
    private boolean cutOffTest(MiniMaxState miniMaxState) {
        return miniMaxState.getStateLevel() == CUT_OFF_THRESHOLD ||
                miniMaxState.getTurnNumber() == this.utils.getAgent().getMaxTurnCount();
    }

    /*
        the state we have here is a state on a leaf node.
     */
    private MiniMaxState eval(MiniMaxState miniMaxState) {
        int maxPlayerScoreUntilHere = miniMaxState.getMaxPlayerScore();
        int minPlayerScoreUntilHere = miniMaxState.getMinPlayerScore();
        int maxPlayerHeuristicScore = 0;
        int minPlayerHeuristicScore = 0;

        if (miniMaxState.getMaxPlayerTile() != null) {
            ComplexBFSNode maxPlayerHeuristic = this.orderingAnalyzer.getOptimalOrderForAdversarialAnalyzer(
                    miniMaxState.getMaxPlayerTile(), miniMaxState.getGridOnThisState(), miniMaxState.getMaxPlayerCollectedGems(),
                    miniMaxState.getMaxPlayerScore(), miniMaxState.getTurnNumber()
            );
            if (maxPlayerHeuristic != null) {
                maxPlayerHeuristicScore = maxPlayerHeuristic.getRewardUntilHere() - maxPlayerHeuristic.getCostUntilHere();
            }
        }
        if (miniMaxState.getMinPlayerTile() != null) {
            ComplexBFSNode minPlayerHeuristic = this.orderingAnalyzer.getOptimalOrderForAdversarialAnalyzer(
                    miniMaxState.getMinPlayerTile(), miniMaxState.getGridOnThisState(), miniMaxState.getMinPlayerCollectedGems(),
                    miniMaxState.getMinPlayerScore(), miniMaxState.getTurnNumber()
            );
            if (minPlayerHeuristic != null) {
                minPlayerHeuristicScore = minPlayerHeuristic.getRewardUntilHere() - minPlayerHeuristic.getCostUntilHere();
            }
        }

        int maxPlayerTotalScores = maxPlayerScoreUntilHere + maxPlayerHeuristicScore;
        int minPlayerTotalScores = minPlayerScoreUntilHere + minPlayerHeuristicScore;

        miniMaxState.setDiffScores(maxPlayerTotalScores - minPlayerTotalScores);
        return miniMaxState;
    }

    private MiniMaxState getMaxState(MiniMaxState miniMaxState1, MiniMaxState miniMaxState2) {
        int diffScoresOfState1 = miniMaxState1.getDiffScores();
        int diffScoresOfState2 = miniMaxState2.getDiffScores();
        if (diffScoresOfState1 >= diffScoresOfState2) {
            return miniMaxState1;
        } else {
            return miniMaxState2;
        }
    }

    private MiniMaxState getMinState(MiniMaxState miniMaxState1, MiniMaxState miniMaxState2) {
        int diffScoresOfState1 = miniMaxState1.getDiffScores();
        int diffScoresOfState2 = miniMaxState2.getDiffScores();
        if (diffScoresOfState1 >= diffScoresOfState2) {
            return miniMaxState2;
        } else {
            return miniMaxState1;
        }
    }

    /*
        1. the state we receive here is a state in middle nodes
        and we output a state related to a leaf node or
        another middle node.

        2. teleport action considers in another function because
        it may have multiple results.

        3. validation of input actions are checked before.
     */
    private MiniMaxState resultForActionsExceptTeleport(MiniMaxState miniMaxState, BaseAgent.Action action, PlayerType playerType) {
        if (action.equals(BaseAgent.Action.Left)) {
            if (playerType.equals(PlayerType.MAX_PLAYER)) {
                Tile maxPlayerTile = miniMaxState.getMaxPlayerTile();
                Tile leftTile = new Tile(maxPlayerTile.getX(), maxPlayerTile.getY() - 1,
                        this.utils.getTileType(maxPlayerTile.getX(), maxPlayerTile.getY() - 1, miniMaxState.getGridOnThisState()));
                return moveActionResultForMaxPlayer(miniMaxState, action, leftTile);
            } else {
                Tile minPlayerTile = miniMaxState.getMinPlayerTile();
                Tile leftTile = new Tile(minPlayerTile.getX(), minPlayerTile.getY() - 1,
                        this.utils.getTileType(minPlayerTile.getX(), minPlayerTile.getY() - 1, miniMaxState.getGridOnThisState()));
                return moveActionResultForMinPlayer(miniMaxState, action, leftTile);
            }
        } else if (action.equals(BaseAgent.Action.Up)) {
            if (playerType.equals(PlayerType.MAX_PLAYER)) {
                Tile maxPlayerTile = miniMaxState.getMaxPlayerTile();
                Tile upTile = new Tile(maxPlayerTile.getX() - 1, maxPlayerTile.getY(),
                        this.utils.getTileType(maxPlayerTile.getX() - 1, maxPlayerTile.getY(), miniMaxState.getGridOnThisState()));
                return moveActionResultForMaxPlayer(miniMaxState, action, upTile);
            } else {
                Tile minPlayerTile = miniMaxState.getMinPlayerTile();
                Tile upTile = new Tile(minPlayerTile.getX() - 1, minPlayerTile.getY(),
                        this.utils.getTileType(minPlayerTile.getX() - 1, minPlayerTile.getY(), miniMaxState.getGridOnThisState()));
                return moveActionResultForMinPlayer(miniMaxState, action, upTile);
            }
        } else if (action.equals(BaseAgent.Action.Right)) {
            if (playerType.equals(PlayerType.MAX_PLAYER)) {
                Tile maxPlayerTile = miniMaxState.getMaxPlayerTile();
                Tile rightTile = new Tile(maxPlayerTile.getX(), maxPlayerTile.getY() + 1,
                        this.utils.getTileType(maxPlayerTile.getX(), maxPlayerTile.getY() + 1, miniMaxState.getGridOnThisState()));
                return moveActionResultForMaxPlayer(miniMaxState, action, rightTile);
            } else {
                Tile minPlayerTile = miniMaxState.getMinPlayerTile();
                Tile rightTile = new Tile(minPlayerTile.getX(), minPlayerTile.getY() + 1,
                        this.utils.getTileType(minPlayerTile.getX(), minPlayerTile.getY() + 1, miniMaxState.getGridOnThisState()));
                return moveActionResultForMinPlayer(miniMaxState, action, rightTile);
            }
        } else if (action.equals(BaseAgent.Action.Down)) {
            if (playerType.equals(PlayerType.MAX_PLAYER)) {
                Tile maxPlayerTile = miniMaxState.getMaxPlayerTile();
                Tile downTile = new Tile(maxPlayerTile.getX() + 1, maxPlayerTile.getY(),
                        this.utils.getTileType(maxPlayerTile.getX() + 1, maxPlayerTile.getY(), miniMaxState.getGridOnThisState()));
                return moveActionResultForMaxPlayer(miniMaxState, action, downTile);
            } else {
                Tile minPlayerTile = miniMaxState.getMinPlayerTile();
                Tile downTile = new Tile(minPlayerTile.getX() + 1, minPlayerTile.getY(),
                        this.utils.getTileType(minPlayerTile.getX() + 1, minPlayerTile.getY(), miniMaxState.getGridOnThisState()));
                return moveActionResultForMinPlayer(miniMaxState, action, downTile);
            }
        } else if (action.equals(BaseAgent.Action.Trap)) {
            return trapActionResultForMaxPlayer(miniMaxState, action);
        } else if (action.equals(BaseAgent.Action.NoOp)) {
            if (playerType.equals(PlayerType.MAX_PLAYER))
                return noopActionResultForMaxPlayer(miniMaxState, action);
            else
                return noopActionResultForMinPlayer(miniMaxState, action);
        }
        throw new IllegalStateException("actions cannot have no result!");
    }

    private List<MiniMaxState> resultsForTeleportAction(MiniMaxState miniMaxState, BaseAgent.Action action, PlayerType playerType) {
        if (action != BaseAgent.Action.Teleport)
            throw new IllegalStateException("we can only return result of teleport for teleport action!");
        List<MiniMaxState> results = new LinkedList<>();
        List<Tile> teleportTiles = this.utils.getTileTypeListMap().get(TileType.TELEPORT);
        for (Tile teleport : teleportTiles) {
            Tile sourceTile;
            if (playerType.equals(PlayerType.MAX_PLAYER)) {
                sourceTile = miniMaxState.getMaxPlayerTile();
                if (!teleport.equals(sourceTile)) {
                    results.add(teleportActionResultForMaxPlayer(miniMaxState, action, teleport));
                }
            } else {
                sourceTile = miniMaxState.getMinPlayerTile();
                if (!teleport.equals(sourceTile)) {
                    results.add(teleportActionResultForMinPlayer(miniMaxState, action, teleport));
                }
            }

        }
        return results;
    }

    private MiniMaxState teleportActionResultForMaxPlayer(MiniMaxState miniMaxState, BaseAgent.Action action, Tile destination) {
        // defining state properties
        List<BaseAgent.Action> newActionsDoneUntilHere = new LinkedList<>(miniMaxState.getActionsDoneUntilHere());
        newActionsDoneUntilHere.add(action);
        String[][] newGridOnThisState = this.utils.gridDeepCopy(miniMaxState.getGridOnThisState());
        Tile oldMaxPlayerTile = this.utils.copyTile(miniMaxState.getMaxPlayerTile());
        int newMaxPlayerScore = miniMaxState.getMaxPlayerScore() - this.utils.getTeleportCost();
        HashMap<TileType, Integer> newMaxPlayerCollectedGems = this.utils.hashMapDeepCopy(miniMaxState.getMaxPlayerCollectedGems());
        int newMaxPlayerTrapUsed = miniMaxState.getMaxPlayerTrapUsed();
        Tile newMinPlayerTile = this.utils.copyTile(miniMaxState.getMinPlayerTile());
        int newMinPlayerScore = miniMaxState.getMinPlayerScore();
        HashMap<TileType, Integer> newMinPlayerCollectedGems = this.utils.hashMapDeepCopy(miniMaxState.getMinPlayerCollectedGems());
        int newStateLevel = miniMaxState.getStateLevel() + 1;
        int newTurnNumber = miniMaxState.getTurnNumber() + (newStateLevel % 2);

        String maxPlayerChar = this.utils.getMyAgentChar();
        String destinationStringInGrid = miniMaxState.getGridOnThisState()[destination.getX()][destination.getY()];
        String sourceStringInGrid = miniMaxState.getGridOnThisState()[oldMaxPlayerTile.getX()][oldMaxPlayerTile.getY()];

        // if collision occurs
        if (destinationStringInGrid.contains(this.utils.getOpponentCharBySource(maxPlayerChar))) {
            /*
                1. grid remains as before both if there is no gems in source tile or
                there is gem in source, because the agent score which is attacking
                won't increase in collision.
                2. in addition to normal properties the only things that may change
                are agents scores.
             */
            if (miniMaxState.getMaxPlayerScore() >= miniMaxState.getMinPlayerScore()) {
                newMinPlayerScore -= HIT_DAMAGE;
            } else {
                newMaxPlayerScore -= HIT_DAMAGE;
            }
            return MiniMaxState.builder()
                    .actionsDoneUntilHere(newActionsDoneUntilHere)
                    .gridOnThisState(newGridOnThisState)
                    .maxPlayerTile(oldMaxPlayerTile)
                    .maxPlayerScore(newMaxPlayerScore)
                    .maxPlayerCollectedGems(newMaxPlayerCollectedGems)
                    .maxPlayerTrapUsed(newMaxPlayerTrapUsed)
                    .minPlayerTile(newMinPlayerTile)
                    .minPlayerScore(newMinPlayerScore)
                    .minPlayerCollectedGems(newMinPlayerCollectedGems)
                    .stateLevel(newStateLevel)
                    .turnNumber(newTurnNumber)
                    .build();
        }
        // if destination is teleport. in fact it should be teleport.
        if (destinationStringInGrid.contains(TileType.TELEPORT.getValue())) {
            newGridOnThisState[oldMaxPlayerTile.getX()][oldMaxPlayerTile.getY()] =
                    updateSourceTileStringWhenAgentLeavesThere(sourceStringInGrid);
            newGridOnThisState[destination.getX()][destination.getY()] =
                    updateDestinationTileStringWhenAgentEntersThere(destinationStringInGrid, maxPlayerChar);
            Tile newMaxPlayerTile = this.utils.copyTile(destination);
            return MiniMaxState.builder()
                    .actionsDoneUntilHere(newActionsDoneUntilHere)
                    .gridOnThisState(newGridOnThisState)
                    .maxPlayerTile(newMaxPlayerTile)
                    .maxPlayerScore(newMaxPlayerScore)
                    .maxPlayerCollectedGems(newMaxPlayerCollectedGems)
                    .maxPlayerTrapUsed(newMaxPlayerTrapUsed)
                    .minPlayerTile(newMinPlayerTile)
                    .minPlayerScore(newMinPlayerScore)
                    .minPlayerCollectedGems(newMinPlayerCollectedGems)
                    .stateLevel(newStateLevel)
                    .turnNumber(newTurnNumber)
                    .build();
        }
        throw new IllegalStateException("destination of a teleport should be teleport!");
    }

    private MiniMaxState teleportActionResultForMinPlayer(MiniMaxState miniMaxState, BaseAgent.Action action, Tile destination) {
        // defining state properties
        List<BaseAgent.Action> newActionsDoneUntilHere = new LinkedList<>(miniMaxState.getActionsDoneUntilHere());
        newActionsDoneUntilHere.add(action);
        String[][] newGridOnThisState = this.utils.gridDeepCopy(miniMaxState.getGridOnThisState());
        Tile newMaxPlayerTile = this.utils.copyTile(miniMaxState.getMaxPlayerTile());
        int newMaxPlayerScore = miniMaxState.getMaxPlayerScore();
        HashMap<TileType, Integer> newMaxPlayerCollectedGems = this.utils.hashMapDeepCopy(miniMaxState.getMaxPlayerCollectedGems());
        int newMaxPlayerTrapUsed = miniMaxState.getMaxPlayerTrapUsed();
        Tile oldMinPlayerTile = this.utils.copyTile(miniMaxState.getMinPlayerTile());
        int newMinPlayerScore = miniMaxState.getMinPlayerScore() - this.utils.getTeleportCost();
        HashMap<TileType, Integer> newMinPlayerCollectedGems = this.utils.hashMapDeepCopy(miniMaxState.getMinPlayerCollectedGems());
        int newStateLevel = miniMaxState.getStateLevel() + 1;
        int newTurnNumber = miniMaxState.getTurnNumber() + (newStateLevel % 2);

        String minPlayerChar = this.utils.getOpponentChar();
        String destinationStringInGrid = miniMaxState.getGridOnThisState()[destination.getX()][destination.getY()];
        String sourceStringInGrid = miniMaxState.getGridOnThisState()[oldMinPlayerTile.getX()][oldMinPlayerTile.getY()];

        // if collision occurs
        if (destinationStringInGrid.contains(this.utils.getOpponentCharBySource(minPlayerChar))) {
            /*
                1. grid remains as before both if there is no gems in source tile or
                there is gem in source, because the agent score which is attacking
                won't increase in collision.
                2. in addition to normal properties the only things that may change
                are agents scores.
             */
            if (miniMaxState.getMinPlayerScore() >= miniMaxState.getMaxPlayerScore()) {
                newMaxPlayerScore -= HIT_DAMAGE;
            } else {
                newMinPlayerScore -= HIT_DAMAGE;
            }
            return MiniMaxState.builder()
                    .actionsDoneUntilHere(newActionsDoneUntilHere)
                    .gridOnThisState(newGridOnThisState)
                    .maxPlayerTile(newMaxPlayerTile)
                    .maxPlayerScore(newMaxPlayerScore)
                    .maxPlayerCollectedGems(newMaxPlayerCollectedGems)
                    .maxPlayerTrapUsed(newMaxPlayerTrapUsed)
                    .minPlayerTile(oldMinPlayerTile)
                    .minPlayerScore(newMinPlayerScore)
                    .minPlayerCollectedGems(newMinPlayerCollectedGems)
                    .stateLevel(newStateLevel)
                    .turnNumber(newTurnNumber)
                    .build();
        }
        // if destination teleport. in fact it should be teleport.
        if (destinationStringInGrid.contains(TileType.TELEPORT.getValue())) {
            newGridOnThisState[oldMinPlayerTile.getX()][oldMinPlayerTile.getY()] =
                    updateSourceTileStringWhenAgentLeavesThere(sourceStringInGrid);
            newGridOnThisState[destination.getX()][destination.getY()] =
                    updateDestinationTileStringWhenAgentEntersThere(destinationStringInGrid, minPlayerChar);
            Tile newMinPlayerTile = this.utils.copyTile(destination);
            return MiniMaxState.builder()
                    .actionsDoneUntilHere(newActionsDoneUntilHere)
                    .gridOnThisState(newGridOnThisState)
                    .maxPlayerTile(newMaxPlayerTile)
                    .maxPlayerScore(newMaxPlayerScore)
                    .maxPlayerCollectedGems(newMaxPlayerCollectedGems)
                    .maxPlayerTrapUsed(newMaxPlayerTrapUsed)
                    .minPlayerTile(newMinPlayerTile)
                    .minPlayerScore(newMinPlayerScore)
                    .minPlayerCollectedGems(newMinPlayerCollectedGems)
                    .stateLevel(newStateLevel)
                    .turnNumber(newTurnNumber)
                    .build();
        }
        throw new IllegalStateException("destination of a teleport should be teleport!");
    }

    /*
        1. validation of movement is checked before
        2. destination is not wall
        3. destination does not contain opponent trap char
        but can contains my agent trap char
        4. opponent can be in destination which makes a collision happen
     */
    private MiniMaxState moveActionResultForMaxPlayer(MiniMaxState miniMaxState, BaseAgent.Action action, Tile destination) {
        // defining state properties
        List<BaseAgent.Action> newActionsDoneUntilHere = new LinkedList<>(miniMaxState.getActionsDoneUntilHere());
        newActionsDoneUntilHere.add(action);
        String[][] newGridOnThisState = this.utils.gridDeepCopy(miniMaxState.getGridOnThisState());
        Tile oldMaxPlayerTile = this.utils.copyTile(miniMaxState.getMaxPlayerTile());
        int newMaxPlayerScore = miniMaxState.getMaxPlayerScore() - TURN_COST;
        HashMap<TileType, Integer> newMaxPlayerCollectedGems = this.utils.hashMapDeepCopy(miniMaxState.getMaxPlayerCollectedGems());
        int newMaxPlayerTrapUsed = miniMaxState.getMaxPlayerTrapUsed();
        Tile newMinPlayerTile = this.utils.copyTile(miniMaxState.getMinPlayerTile());
        int newMinPlayerScore = miniMaxState.getMinPlayerScore();
        HashMap<TileType, Integer> newMinPlayerCollectedGems = this.utils.hashMapDeepCopy(miniMaxState.getMinPlayerCollectedGems());
        int newStateLevel = miniMaxState.getStateLevel() + 1;
        int newTurnNumber = miniMaxState.getTurnNumber() + (newStateLevel % 2);

        String maxPlayerChar = this.utils.getMyAgentChar();
        String destinationStringInGrid = miniMaxState.getGridOnThisState()[destination.getX()][destination.getY()];
        String sourceStringInGrid = miniMaxState.getGridOnThisState()[oldMaxPlayerTile.getX()][oldMaxPlayerTile.getY()];

        // if collision occurs
        if (destinationStringInGrid.contains(this.utils.getOpponentCharBySource(maxPlayerChar))) {
            /*
                1. grid remains as before both if there is no gems in source tile or
                there is gem in source, because the agent score which is attacking
                won't increase in collision.
                2. in addition to normal properties the only things that may change
                are agents scores.
             */
            if (miniMaxState.getMaxPlayerScore() >= miniMaxState.getMinPlayerScore()) {
                newMinPlayerScore -= HIT_DAMAGE;
            } else {
                newMaxPlayerScore -= HIT_DAMAGE;
            }
            return MiniMaxState.builder()
                    .actionsDoneUntilHere(newActionsDoneUntilHere)
                    .gridOnThisState(newGridOnThisState)
                    .maxPlayerTile(oldMaxPlayerTile)
                    .maxPlayerScore(newMaxPlayerScore)
                    .maxPlayerCollectedGems(newMaxPlayerCollectedGems)
                    .maxPlayerTrapUsed(newMaxPlayerTrapUsed)
                    .minPlayerTile(newMinPlayerTile)
                    .minPlayerScore(newMinPlayerScore)
                    .minPlayerCollectedGems(newMinPlayerCollectedGems)
                    .stateLevel(newStateLevel)
                    .turnNumber(newTurnNumber)
                    .build();
        }
        // if destination is empty or teleport
        if (destinationStringInGrid.contains(TileType.EMPTY.getValue()) ||
                destinationStringInGrid.contains(TileType.TELEPORT.getValue())) {
            newGridOnThisState[oldMaxPlayerTile.getX()][oldMaxPlayerTile.getY()] =
                    updateSourceTileStringWhenAgentLeavesThere(sourceStringInGrid);
            newGridOnThisState[destination.getX()][destination.getY()] =
                    updateDestinationTileStringWhenAgentEntersThere(destinationStringInGrid, maxPlayerChar);
            Tile newMaxPlayerTile = this.utils.copyTile(destination);
            return MiniMaxState.builder()
                    .actionsDoneUntilHere(newActionsDoneUntilHere)
                    .gridOnThisState(newGridOnThisState)
                    .maxPlayerTile(newMaxPlayerTile)
                    .maxPlayerScore(newMaxPlayerScore)
                    .maxPlayerCollectedGems(newMaxPlayerCollectedGems)
                    .maxPlayerTrapUsed(newMaxPlayerTrapUsed)
                    .minPlayerTile(newMinPlayerTile)
                    .minPlayerScore(newMinPlayerScore)
                    .minPlayerCollectedGems(newMinPlayerCollectedGems)
                    .stateLevel(newStateLevel)
                    .turnNumber(newTurnNumber)
                    .build();
        }
        // if destination is gem
        if (this.utils.isTileTypeOfGemAccordingToStringValue(destinationStringInGrid)) {
            // first we should consider if it is possible to get gem
            if (this.utils.possibleToGetGemByHeuristicDistance(oldMaxPlayerTile, destination, miniMaxState.getMaxPlayerCollectedGems(), miniMaxState.getMaxPlayerScore())) {
                Tile newMaxPlayerTile = new Tile(destination.getX(), destination.getY(), TileType.EMPTY);
                TileType gemType = this.utils.getGemTypeByTileStringValue(destinationStringInGrid);
                newGridOnThisState[oldMaxPlayerTile.getX()][oldMaxPlayerTile.getY()] =
                        updateSourceTileStringWhenAgentLeavesThere(sourceStringInGrid);
                newGridOnThisState[destination.getX()][destination.getY()] =
                        updateDestinationTileStringWhenAgentEntersThereAndPicksGem(destinationStringInGrid, maxPlayerChar);
                newMaxPlayerCollectedGems.replace(gemType, newMaxPlayerCollectedGems.get(gemType) + 1);
                newMaxPlayerScore += this.utils.getGemRewardByType(gemType);
                return MiniMaxState.builder()
                        .actionsDoneUntilHere(newActionsDoneUntilHere)
                        .gridOnThisState(newGridOnThisState)
                        .maxPlayerTile(newMaxPlayerTile)
                        .maxPlayerScore(newMaxPlayerScore)
                        .maxPlayerCollectedGems(newMaxPlayerCollectedGems)
                        .maxPlayerTrapUsed(newMaxPlayerTrapUsed)
                        .minPlayerTile(newMinPlayerTile)
                        .minPlayerScore(newMinPlayerScore)
                        .minPlayerCollectedGems(newMinPlayerCollectedGems)
                        .stateLevel(newStateLevel)
                        .turnNumber(newTurnNumber)
                        .build();
            } else {
                Tile newMaxPlayerTile = this.utils.copyTile(destination);
                newGridOnThisState[oldMaxPlayerTile.getX()][oldMaxPlayerTile.getY()] =
                        updateSourceTileStringWhenAgentLeavesThere(sourceStringInGrid);
                newGridOnThisState[destination.getX()][destination.getY()] =
                        updateDestinationTileStringWhenAgentEntersThere(destinationStringInGrid, maxPlayerChar);
                return MiniMaxState.builder()
                        .actionsDoneUntilHere(newActionsDoneUntilHere)
                        .gridOnThisState(newGridOnThisState)
                        .maxPlayerTile(newMaxPlayerTile)
                        .maxPlayerScore(newMaxPlayerScore)
                        .maxPlayerCollectedGems(newMaxPlayerCollectedGems)
                        .maxPlayerTrapUsed(newMaxPlayerTrapUsed)
                        .minPlayerTile(newMinPlayerTile)
                        .minPlayerScore(newMinPlayerScore)
                        .minPlayerCollectedGems(newMinPlayerCollectedGems)
                        .stateLevel(newStateLevel)
                        .turnNumber(newTurnNumber)
                        .build();
            }
        }
        throw new IllegalStateException("move action cannot have no result for max player!");
    }

    /*
        this function is just like moveActionResultForMaxPlayer except a few
        differences like opposite changes of scores, source and destination.
     */
    private MiniMaxState moveActionResultForMinPlayer(MiniMaxState miniMaxState, BaseAgent.Action action, Tile destination) {
        // defining state properties
        List<BaseAgent.Action> newActionsDoneUntilHere = new LinkedList<>(miniMaxState.getActionsDoneUntilHere());
        newActionsDoneUntilHere.add(action);
        String[][] newGridOnThisState = this.utils.gridDeepCopy(miniMaxState.getGridOnThisState());
        Tile newMaxPlayerTile = this.utils.copyTile(miniMaxState.getMaxPlayerTile());
        int newMaxPlayerScore = miniMaxState.getMaxPlayerScore();
        HashMap<TileType, Integer> newMaxPlayerCollectedGems = this.utils.hashMapDeepCopy(miniMaxState.getMaxPlayerCollectedGems());
        int newMaxPlayerTrapUsed = miniMaxState.getMaxPlayerTrapUsed();
        Tile oldMinPlayerTile = this.utils.copyTile(miniMaxState.getMinPlayerTile());
        int newMinPlayerScore = miniMaxState.getMinPlayerScore() - TURN_COST;
        HashMap<TileType, Integer> newMinPlayerCollectedGems = this.utils.hashMapDeepCopy(miniMaxState.getMinPlayerCollectedGems());
        int newStateLevel = miniMaxState.getStateLevel() + 1;
        int newTurnNumber = miniMaxState.getTurnNumber() + (newStateLevel % 2);

        String minPlayerChar = this.utils.getOpponentChar();
        String destinationStringInGrid = miniMaxState.getGridOnThisState()[destination.getX()][destination.getY()];
        String sourceStringInGrid = miniMaxState.getGridOnThisState()[oldMinPlayerTile.getX()][oldMinPlayerTile.getY()];

        // if collision occurs
        if (destinationStringInGrid.contains(this.utils.getOpponentCharBySource(minPlayerChar))) {
            /*
                1. grid remains as before both if there is no gems in source tile or
                there is gem in source, because the agent score which is attacking
                won't increase in collision.
                2. in addition to normal properties the only things that may change
                are agents scores.
             */
            if (miniMaxState.getMinPlayerScore() >= miniMaxState.getMaxPlayerScore()) {
                newMaxPlayerScore -= HIT_DAMAGE;
            } else {
                newMinPlayerScore -= HIT_DAMAGE;
            }
            return MiniMaxState.builder()
                    .actionsDoneUntilHere(newActionsDoneUntilHere)
                    .gridOnThisState(newGridOnThisState)
                    .maxPlayerTile(newMaxPlayerTile)
                    .maxPlayerScore(newMaxPlayerScore)
                    .maxPlayerCollectedGems(newMaxPlayerCollectedGems)
                    .maxPlayerTrapUsed(newMaxPlayerTrapUsed)
                    .minPlayerTile(oldMinPlayerTile)
                    .minPlayerScore(newMinPlayerScore)
                    .minPlayerCollectedGems(newMinPlayerCollectedGems)
                    .stateLevel(newStateLevel)
                    .turnNumber(newTurnNumber)
                    .build();
        }
        // if destination is empty or teleport
        if (destinationStringInGrid.contains(TileType.EMPTY.getValue()) ||
                destinationStringInGrid.contains(TileType.TELEPORT.getValue())) {
            newGridOnThisState[oldMinPlayerTile.getX()][oldMinPlayerTile.getY()] =
                    updateSourceTileStringWhenAgentLeavesThere(sourceStringInGrid);
            newGridOnThisState[destination.getX()][destination.getY()] =
                    updateDestinationTileStringWhenAgentEntersThere(destinationStringInGrid, minPlayerChar);
            Tile newMinPlayerTile = this.utils.copyTile(destination);
            /*
                check if there is trap in destination.
                note that the destination should not be teleport.
             */
            if (destinationStringInGrid.contains(this.utils.getOpponentTrapCharBySource(minPlayerChar))) {
                newGridOnThisState[destination.getX()][destination.getY()] =
                        updateDestinationTileStringWhenMinPlayerGetInTrap(destinationStringInGrid);
                newMinPlayerScore -= TRAP_DAMAGE;
            }
            return MiniMaxState.builder()
                    .actionsDoneUntilHere(newActionsDoneUntilHere)
                    .gridOnThisState(newGridOnThisState)
                    .maxPlayerTile(newMaxPlayerTile)
                    .maxPlayerScore(newMaxPlayerScore)
                    .maxPlayerCollectedGems(newMaxPlayerCollectedGems)
                    .maxPlayerTrapUsed(newMaxPlayerTrapUsed)
                    .minPlayerTile(newMinPlayerTile)
                    .minPlayerScore(newMinPlayerScore)
                    .minPlayerCollectedGems(newMinPlayerCollectedGems)
                    .stateLevel(newStateLevel)
                    .turnNumber(newTurnNumber)
                    .build();
        }
        // if destination is gem
        if (this.utils.isTileTypeOfGemAccordingToStringValue(destinationStringInGrid)) {
            // first we should consider if it is possible to get gem
            if (this.utils.possibleToGetGemByHeuristicDistance(oldMinPlayerTile, destination, miniMaxState.getMinPlayerCollectedGems(), miniMaxState.getMinPlayerScore())) {
                Tile newMinPlayerTile = new Tile(destination.getX(), destination.getY(), TileType.EMPTY);
                TileType gemType = this.utils.getGemTypeByTileStringValue(destinationStringInGrid);
                newGridOnThisState[oldMinPlayerTile.getX()][oldMinPlayerTile.getY()] =
                        updateSourceTileStringWhenAgentLeavesThere(sourceStringInGrid);
                newGridOnThisState[destination.getX()][destination.getY()] =
                        updateDestinationTileStringWhenAgentEntersThereAndPicksGem(destinationStringInGrid, minPlayerChar);
                newMinPlayerCollectedGems.replace(gemType, newMinPlayerCollectedGems.get(gemType) + 1);
                newMinPlayerScore += this.utils.getGemRewardByType(gemType);
                /*
                check if there is trap in destination.
                note that the destination should not be teleport.
                */
                if (destinationStringInGrid.contains(this.utils.getOpponentTrapCharBySource(minPlayerChar))) {
                    newGridOnThisState[destination.getX()][destination.getY()] =
                            updateDestinationTileStringWhenMinPlayerGetInTrap(destinationStringInGrid);
                    newMinPlayerScore -= TRAP_DAMAGE;
                }
                return MiniMaxState.builder()
                        .actionsDoneUntilHere(newActionsDoneUntilHere)
                        .gridOnThisState(newGridOnThisState)
                        .maxPlayerTile(newMaxPlayerTile)
                        .maxPlayerScore(newMaxPlayerScore)
                        .maxPlayerCollectedGems(newMaxPlayerCollectedGems)
                        .maxPlayerTrapUsed(newMaxPlayerTrapUsed)
                        .minPlayerTile(newMinPlayerTile)
                        .minPlayerScore(newMinPlayerScore)
                        .minPlayerCollectedGems(newMinPlayerCollectedGems)
                        .stateLevel(newStateLevel)
                        .turnNumber(newTurnNumber)
                        .build();
            } else {
                Tile newMinPlayerTile = this.utils.copyTile(destination);
                newGridOnThisState[oldMinPlayerTile.getX()][oldMinPlayerTile.getY()] =
                        updateSourceTileStringWhenAgentLeavesThere(sourceStringInGrid);
                newGridOnThisState[destination.getX()][destination.getY()] =
                        updateDestinationTileStringWhenAgentEntersThere(destinationStringInGrid, minPlayerChar);
                return MiniMaxState.builder()
                        .actionsDoneUntilHere(newActionsDoneUntilHere)
                        .gridOnThisState(newGridOnThisState)
                        .maxPlayerTile(newMaxPlayerTile)
                        .maxPlayerScore(newMaxPlayerScore)
                        .maxPlayerCollectedGems(newMaxPlayerCollectedGems)
                        .maxPlayerTrapUsed(newMaxPlayerTrapUsed)
                        .minPlayerTile(newMinPlayerTile)
                        .minPlayerScore(newMinPlayerScore)
                        .minPlayerCollectedGems(newMinPlayerCollectedGems)
                        .stateLevel(newStateLevel)
                        .turnNumber(newTurnNumber)
                        .build();
            }
        }
        throw new IllegalStateException("move action cannot have no result for min player!");
    }

    private MiniMaxState trapActionResultForMaxPlayer(MiniMaxState miniMaxState, BaseAgent.Action action) {
        // defining state properties
        List<BaseAgent.Action> newActionsDoneUntilHere = new LinkedList<>(miniMaxState.getActionsDoneUntilHere());
        newActionsDoneUntilHere.add(action);
        String[][] newGridOnThisState = this.utils.gridDeepCopy(miniMaxState.getGridOnThisState());
        Tile newMaxPlayerTile = this.utils.copyTile(miniMaxState.getMaxPlayerTile());
        int newMaxPlayerScore = miniMaxState.getMaxPlayerScore() - TURN_COST;
        HashMap<TileType, Integer> newMaxPlayerCollectedGems = this.utils.hashMapDeepCopy(miniMaxState.getMaxPlayerCollectedGems());
        int newMaxPlayerTrapUsed = miniMaxState.getMaxPlayerTrapUsed() + 1;
        Tile newMinPlayerTile = this.utils.copyTile(miniMaxState.getMinPlayerTile());
        int newMinPlayerScore = miniMaxState.getMinPlayerScore();
        HashMap<TileType, Integer> newMinPlayerCollectedGems = this.utils.hashMapDeepCopy(miniMaxState.getMinPlayerCollectedGems());
        int newStateLevel = miniMaxState.getStateLevel() + 1;
        int newTurnNumber = miniMaxState.getTurnNumber() + (newStateLevel % 2);

        String sourceStringInGrid = miniMaxState.getGridOnThisState()[newMaxPlayerTile.getX()][newMaxPlayerTile.getY()];
        newGridOnThisState[newMaxPlayerTile.getX()][newMaxPlayerTile.getY()] =
                updateSourceTileStringWhenAgentTrapsThere(sourceStringInGrid);
        return MiniMaxState.builder()
                .actionsDoneUntilHere(newActionsDoneUntilHere)
                .gridOnThisState(newGridOnThisState)
                .maxPlayerTile(newMaxPlayerTile)
                .maxPlayerScore(newMaxPlayerScore)
                .maxPlayerCollectedGems(newMaxPlayerCollectedGems)
                .maxPlayerTrapUsed(newMaxPlayerTrapUsed)
                .minPlayerTile(newMinPlayerTile)
                .minPlayerScore(newMinPlayerScore)
                .minPlayerCollectedGems(newMinPlayerCollectedGems)
                .stateLevel(newStateLevel)
                .turnNumber(newTurnNumber)
                .build();
    }

    private MiniMaxState noopActionResultForMaxPlayer(MiniMaxState miniMaxState, BaseAgent.Action action) {
        // defining state properties
        List<BaseAgent.Action> newActionsDoneUntilHere = new LinkedList<>(miniMaxState.getActionsDoneUntilHere());
        newActionsDoneUntilHere.add(action);
        String[][] newGridOnThisState = this.utils.gridDeepCopy(miniMaxState.getGridOnThisState());
        Tile newMaxPlayerTile = this.utils.copyTile(miniMaxState.getMaxPlayerTile());
        int newMaxPlayerScore = miniMaxState.getMaxPlayerScore() - TURN_COST;
        HashMap<TileType, Integer> newMaxPlayerCollectedGems = this.utils.hashMapDeepCopy(miniMaxState.getMaxPlayerCollectedGems());
        int newMaxPlayerTrapUsed = miniMaxState.getMaxPlayerTrapUsed();
        Tile newMinPlayerTile = this.utils.copyTile(miniMaxState.getMinPlayerTile());
        int newMinPlayerScore = miniMaxState.getMinPlayerScore();
        HashMap<TileType, Integer> newMinPlayerCollectedGems = this.utils.hashMapDeepCopy(miniMaxState.getMinPlayerCollectedGems());
        int newStateLevel = miniMaxState.getStateLevel() + 1;
        int newTurnNumber = miniMaxState.getTurnNumber() + (newStateLevel % 2);

        String sourceStringInGrid = miniMaxState.getGridOnThisState()[newMaxPlayerTile.getX()][newMaxPlayerTile.getY()];
        newGridOnThisState[newMaxPlayerTile.getX()][newMaxPlayerTile.getY()] =
                updateSourceTileStringWhenAgentTrapsThere(sourceStringInGrid);
        return MiniMaxState.builder()
                .actionsDoneUntilHere(newActionsDoneUntilHere)
                .gridOnThisState(newGridOnThisState)
                .maxPlayerTile(newMaxPlayerTile)
                .maxPlayerScore(newMaxPlayerScore)
                .maxPlayerCollectedGems(newMaxPlayerCollectedGems)
                .maxPlayerTrapUsed(newMaxPlayerTrapUsed)
                .minPlayerTile(newMinPlayerTile)
                .minPlayerScore(newMinPlayerScore)
                .minPlayerCollectedGems(newMinPlayerCollectedGems)
                .stateLevel(newStateLevel)
                .turnNumber(newTurnNumber)
                .build();
    }

    private MiniMaxState noopActionResultForMinPlayer(MiniMaxState miniMaxState, BaseAgent.Action action) {
        // defining state properties
        List<BaseAgent.Action> newActionsDoneUntilHere = new LinkedList<>(miniMaxState.getActionsDoneUntilHere());
        newActionsDoneUntilHere.add(action);
        String[][] newGridOnThisState = this.utils.gridDeepCopy(miniMaxState.getGridOnThisState());
        Tile newMaxPlayerTile = this.utils.copyTile(miniMaxState.getMaxPlayerTile());
        int newMaxPlayerScore = miniMaxState.getMaxPlayerScore();
        HashMap<TileType, Integer> newMaxPlayerCollectedGems = this.utils.hashMapDeepCopy(miniMaxState.getMaxPlayerCollectedGems());
        int newMaxPlayerTrapUsed = miniMaxState.getMaxPlayerTrapUsed();
        Tile newMinPlayerTile = this.utils.copyTile(miniMaxState.getMinPlayerTile());
        int newMinPlayerScore = miniMaxState.getMinPlayerScore() - TURN_COST;
        HashMap<TileType, Integer> newMinPlayerCollectedGems = this.utils.hashMapDeepCopy(miniMaxState.getMinPlayerCollectedGems());
        int newStateLevel = miniMaxState.getStateLevel() + 1;
        int newTurnNumber = miniMaxState.getTurnNumber() + (newStateLevel % 2);

        return MiniMaxState.builder()
                .actionsDoneUntilHere(newActionsDoneUntilHere)
                .gridOnThisState(newGridOnThisState)
                .maxPlayerTile(newMaxPlayerTile)
                .maxPlayerScore(newMaxPlayerScore)
                .maxPlayerCollectedGems(newMaxPlayerCollectedGems)
                .maxPlayerTrapUsed(newMaxPlayerTrapUsed)
                .minPlayerTile(newMinPlayerTile)
                .minPlayerScore(newMinPlayerScore)
                .minPlayerCollectedGems(newMinPlayerCollectedGems)
                .stateLevel(newStateLevel)
                .turnNumber(newTurnNumber)
                .build();
    }

    /*
        trap action is not considered in minimax algorithm
     */
    private List<BaseAgent.Action> getPossibleActions(MiniMaxState miniMaxState, Tile player) {
        List<BaseAgent.Action> possibleActions = new LinkedList<>();
        if (this.utils.isTeleportActionValid(player, miniMaxState.getGridOnThisState())) {
            possibleActions.add(BaseAgent.Action.Teleport);
        }
        if (this.utils.isLeftActionValid(player, miniMaxState.getGridOnThisState())) {
            possibleActions.add(BaseAgent.Action.Left);
        }
        if (this.utils.isUpActionValid(player, miniMaxState.getGridOnThisState())) {
            possibleActions.add(BaseAgent.Action.Up);
        }
        if (this.utils.isRightActionValid(player, miniMaxState.getGridOnThisState())) {
            possibleActions.add(BaseAgent.Action.Right);
        }
        if (this.utils.isDownActionValid(player, miniMaxState.getGridOnThisState())) {
            possibleActions.add(BaseAgent.Action.Down);
        }
        /*
            we only consider trap action for our agent
            since opponent agent remained trap is not deterministic
            and considering it may cause optimistic results for opponent agent.
         */
        if (this.utils.getPlayerTypeByTile(miniMaxState.getGridOnThisState(), player).equals(PlayerType.MAX_PLAYER)) {
            if (this.utils.isTrapActionValid(miniMaxState.getMaxPlayerTile(), miniMaxState)) {
                possibleActions.add(BaseAgent.Action.Trap);
            }
        }
//        possibleActions.add(BaseAgent.Action.NoOp);
        return possibleActions;
    }

    private String updateSourceTileStringWhenAgentLeavesThere(String source) {
        StringBuilder stringBuilder = new StringBuilder(source);
        return stringBuilder.deleteCharAt(1).toString();
    }

    private String updateDestinationTileStringWhenAgentEntersThere(String destination, String agentChar) {
        return new StringBuilder(destination).insert(1, agentChar).toString();
    }

    private String updateDestinationTileStringWhenAgentEntersThereAndPicksGem(String destination, String agentChar) {
        return new StringBuilder(destination).replace(0, 1, TileType.EMPTY.getValue()).insert(1, agentChar).toString();
    }

    private String updateSourceTileStringWhenAgentTrapsThere(String source) {
        return source + this.utils.getMyAgentTrapChar();
    }

    private String updateDestinationTileStringWhenMinPlayerGetInTrap(String destination) {
        StringBuilder stringBuilder = new StringBuilder(destination);
        return stringBuilder.deleteCharAt(destination.length() - 1).toString();
    }

}
