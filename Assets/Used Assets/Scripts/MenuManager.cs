using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.Audio;
using TMPro; 

public class MenuManager : MonoBehaviour
{
    public AudioMixer myMixer;

    [Header("Panels")]
    public GameObject mainMenuPanel;
    public GameObject settingsPanel;
    public GameObject highScorePanel; 

    [Header("High Score Display")]
    public TextMeshProUGUI highScoreText;
    public void PlayGame()
    {
        SceneManager.LoadScene(SceneManager.GetActiveScene().buildIndex + 1);
    }

    public void OpenSettings()
    {
        mainMenuPanel.SetActive(false);
        settingsPanel.SetActive(true);
    }

    public void CloseSettings()
    {
        settingsPanel.SetActive(false);
        mainMenuPanel.SetActive(true);
    }



    public void OpenHighScores()
    {
        mainMenuPanel.SetActive(false);
        highScorePanel.SetActive(true);

        // 1. Load the JSON
        string json = PlayerPrefs.GetString("LeaderboardData", "{}");
        HighScoreData data = JsonUtility.FromJson<HighScoreData>(json);

        // 2. Build the display string
        string boardText = "TOP 10 LEGENDS\n\n";

        for (int i = 0; i < data.list.Count; i++)
        {
            boardText += (i + 1) + ". " + data.list[i].initials + " - " + data.list[i].score + "\n";
        }

        // 3. If list is empty, show a placeholder
        if (data.list.Count == 0) boardText = "No scores yet!";

        highScoreText.text = boardText;
    }

    public void CloseHighScores()
    {
        highScorePanel.SetActive(false);
        mainMenuPanel.SetActive(true);
    }

   

    public void SetVolume(float value)
    {
        myMixer.SetFloat("MasterVolume", Mathf.Log10(value) * 20);
    }

    public void QuitGame()
    {
        Application.Quit();
        Debug.Log("Game Exited");
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#endif
    }
}