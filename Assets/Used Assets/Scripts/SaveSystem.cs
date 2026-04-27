using UnityEngine;
using TMPro;
using UnityEngine.SceneManagement;
using System.Collections.Generic; 

public class SaveSystem : MonoBehaviour
{
    public TMP_InputField initialsInput;
    public CharacterController2D playerScript;

    public void SaveScoreAndQuit()
    {
        
        string initials = initialsInput.text.ToUpper();
        if (string.IsNullOrEmpty(initials)) initials = "???"; 

        int scoreToSave = playerScript.totalScore;

        
        string json = PlayerPrefs.GetString("LeaderboardData", "{}");
        HighScoreData data = JsonUtility.FromJson<HighScoreData>(json);

        
        data.list.Add(new HighScoreEntry(initials, scoreToSave));
        data.list.Sort((x, y) => y.score.CompareTo(x.score)); 

        
        if (data.list.Count > 10)
        {
            data.list.RemoveRange(10, data.list.Count - 10);
        }

        
        string newJson = JsonUtility.ToJson(data);
        PlayerPrefs.SetString("LeaderboardData", newJson);
        PlayerPrefs.Save();

        
        Time.timeScale = 1f;
        SceneManager.LoadScene("MainMenu");
    }
}



[System.Serializable]
public class HighScoreEntry
{
    public string initials;
    public int score;

    public HighScoreEntry(string initials, int score)
    {
        this.initials = initials;
        this.score = score;
    }
}

[System.Serializable]
public class HighScoreData
{
    public List<HighScoreEntry> list = new List<HighScoreEntry>();
}