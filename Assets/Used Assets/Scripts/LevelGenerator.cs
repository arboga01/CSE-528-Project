using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Tilemaps;

public class LevelGenerator : MonoBehaviour
{
    [Header("Assign SPRITES (Blue Cube Icons) Here")]
    public Tilemap tilemap;
    public Sprite[] grassSprites;
    public Sprite[] brickSprites;
    public Sprite[] caveSprites;

    [Header("Dimensions")]
    public int width = 100;
    public int height = 50;
    public int brickWidth = 5;

    [Header("Cave Generation Settings")]
    [Range(0, 100)]
    public int fillPercentage = 50;

    // These lists hold the internal "Tile" versions of your sprites
    private List<Tile> grassTiles = new List<Tile>();
    private List<Tile> brickTiles = new List<Tile>();
    private List<Tile> caveTiles = new List<Tile>();

    void Start()
    {
        GenerateLevel();
    }

    [ContextMenu("Generate Level")]
    public void GenerateLevel()
    {
        if (tilemap == null)
        {
            Debug.LogError("Assign the Tilemap in the Inspector first!");
            return;
        }

        // 1. Prepare the tiles
        SetupTiles();

        // 2. Clear the old ones
        tilemap.ClearAllTiles();

        // 3. Build the world
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                // Top layer: Grass
                if (y == height - 1)
                {
                    PlaceTile(x, y, grassTiles);
                }
                // Side walls: Bricks
                else if (x < brickWidth || x >= width - brickWidth)
                {
                    // Opening in the middle
                    if (y < height / 2 - 2 || y > height / 2 + 2)
                    {
                        PlaceTile(x, y, brickTiles);
                    }
                }
                // Middle area: Cave
                else
                {
                    if (Random.Range(0, 100) < fillPercentage)
                    {
                        PlaceTile(x, y, caveTiles);
                    }
                }
            }
        }
    }

    // --- These functions are now outside of GenerateLevel where they belong! ---

    void SetupTiles()
    {
        grassTiles = ConvertToTiles(grassSprites);
        brickTiles = ConvertToTiles(brickSprites);
        caveTiles = ConvertToTiles(caveSprites);
    }

    List<Tile> ConvertToTiles(Sprite[] sprites)
    {
        List<Tile> newList = new List<Tile>();
        if (sprites == null) return newList;

        foreach (Sprite s in sprites)
        {
            if (s == null) continue;
            Tile t = ScriptableObject.CreateInstance<Tile>();
            t.sprite = s;
            newList.Add(t);
        }
        return newList;
    }

    void PlaceTile(int x, int y, List<Tile> tiles)
    {
        if (tiles.Count > 0)
        {
            tilemap.SetTile(new Vector3Int(x, y, 0), tiles[Random.Range(0, tiles.Count)]);
        }
    }

    void CenterCamera()
    {
           Camera mainCam = Camera.main;
        if (mainCam == null) return;

        float centerX = (float)width / 2f;
        float centerY = (float)height / 2f;

        mainCam.transform.position = new Vector3(centerX, centerY, -10);
        mainCam.orthographicSize = (float)height / 2f;
    }
}