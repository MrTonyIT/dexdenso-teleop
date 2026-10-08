PROGRAM main

'!==============================================================================
'! TELEOPERATION BRIDGE SERVER (DENSO RC8 / WINCAPS III)
'!==============================================================================

DIM X! AS SINGLE
DIM Y! AS SINGLE
DIM Z! AS SINGLE
DIM Rx! AS SINGLE
DIM Ry! AS SINGLE
DIM Rz! AS SINGLE
DIM Grip% AS INTEGER

DIM P1 AS POSITION
DIM P_SafeHome AS POSITION
DIM LastGrip% AS INTEGER

ON ERROR GOTO ErrorHandler

ChangeWork 0
ChangeTool 0
Motor On
TAKEARM KEEP = 1

ExtSpeed 100, 100
SPEED 100
ACCEL 100, 100
HighPathAccuracy False

P_SafeHome = P(425.0, 0.0, 325.0, 180.0, 0.0, 0.0, 5)
MOVE P, @E P_SafeHome

LastGrip% = -1

Print "DENSO RC8: Starting TCP Socket Server on Port 5000..."
OPEN "TCP:5000" AS #1
Print "DENSO RC8: Socket Server Ready. Waiting for Python Bridge..."

DO
    INPUT #1, X!, Y!, Z!, Rx!, Ry!, Rz!, Grip%

    IF Grip% <> LastGrip% THEN
        IF Grip% = 1 THEN
            SET IO[128]
            Print "Gripper: CLOSED"
        ELSE
            RESET IO[128]
            Print "Gripper: OPEN"
        END IF
        LastGrip% = Grip%
    END IF

    P1 = P(X!, Y!, Z!, Rx!, Ry!, Rz!, 5)

    IF CurDist > 25.0 THEN
        WAIT (CurDist <= 25.0) OR (MotionComplete = 1)
    END IF

    MOVE P, @P P1, NEXT

LOOP

ErrorHandler:
    Print "DENSO RC8: Error or Disconnected! Code: ", Err
    HOLD
    RESET IO[128]
    CLOSE #1
    GIVEARM
    Print "DENSO RC8: System Released Safely."
END
