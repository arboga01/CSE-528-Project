using UnityEngine;
using TMPro;

public class CharacterController2D : MonoBehaviour
{
    [Header("Movement Stats")]
    public float moveSpeed = 10f;
    public float jumpForce = 15f;

    [Header("Ground Detection")]
    public Transform groundCheck;
    public float checkRadius = 0.2f;
    public LayerMask groundLayer;

    [Header("UI & Collection")]
    public TextMeshProUGUI scoreText;
    public GameObject scoreEntryPanel;
    public float groundLevelY = -44f;
    public float scoreMultiplier = 10f;
    public int totalScore = 0;

    [Header("Audio")]
    public AudioSource sfxSource;
    public AudioSource runSource;
    public AudioClip coinSound;
    public AudioClip jumpSound;
    public AudioClip runSound;

    private Rigidbody2D rb;
    private Animator anim;
    private SpriteRenderer sprite;
    private bool isGrounded;
    private float moveInput;
    private bool isInputtingScore = false;

    void Start()
    {
        rb = GetComponent<Rigidbody2D>();
        anim = GetComponent<Animator>();
        sprite = GetComponent<SpriteRenderer>();

        if (runSource != null && runSound != null)
        {
            runSource.clip = runSound;
            runSource.loop = true;
        }

        if (scoreEntryPanel != null) scoreEntryPanel.SetActive(false);
        UpdateScoreUI();
    }

    void Update()
    {
        if (isInputtingScore) return;

        // Trigger Score Entry with Enter Key
        if (Input.GetKeyDown(KeyCode.Return) || Input.GetKeyDown(KeyCode.KeypadEnter))
        {
            OpenScoreEntry();
        }

        moveInput = Input.GetAxisRaw("Horizontal");

        if (moveInput > 0) sprite.flipX = false;
        else if (moveInput < 0) sprite.flipX = true;

        anim.SetFloat("Speed", Mathf.Abs(moveInput));
        anim.SetBool("isGrounded", isGrounded);
        anim.SetFloat("yVelocity", rb.linearVelocity.y);

        // JUMP LOGIC: Physics always happens, cost is conditional
        if (Input.GetButtonDown("Jump") && isGrounded)
        {
            rb.linearVelocity = new Vector2(rb.linearVelocity.x, jumpForce);
            sfxSource.PlayOneShot(jumpSound);

            if (totalScore > 0)
            {
                totalScore--;
                UpdateScoreUI();
            }
        }

        HandleRunSound(); // Moved back into the main Update loop
    }

    // Now properly separated from the Update method
    void OpenScoreEntry()
    {
        isInputtingScore = true;
        rb.linearVelocity = Vector2.zero;
        anim.SetFloat("Speed", 0);
        if (scoreEntryPanel != null) scoreEntryPanel.SetActive(true);
        Time.timeScale = 0f; // Pause the world for typing
    }

    void FixedUpdate()
    {
        if (isInputtingScore) return;
        rb.linearVelocity = new Vector2(moveInput * moveSpeed, rb.linearVelocity.y);
        isGrounded = Physics2D.OverlapCircle(groundCheck.position, checkRadius, groundLayer);
    }

    private void HandleRunSound()
    {
        if (Mathf.Abs(moveInput) > 0.1f && isGrounded)
        {
            if (!runSource.isPlaying) runSource.Play();
        }
        else
        {
            if (runSource.isPlaying) runSource.Stop();
        }
    }

    private void OnTriggerEnter2D(Collider2D collision)
    {
        if (collision.gameObject.CompareTag("Coin"))
        {
            sfxSource.PlayOneShot(coinSound);

            float coinY = collision.transform.position.y;
            float distanceFromGround = Mathf.Abs(coinY - groundLevelY);
            int coinValue = Mathf.RoundToInt((distanceFromGround * scoreMultiplier) + 1);

            totalScore += coinValue;
            UpdateScoreUI();
            Destroy(collision.gameObject);
        }
    }

    void UpdateScoreUI()
    {
        if (scoreText != null)
        {
            scoreText.text = "Score: " + totalScore.ToString();
        }
    }
}