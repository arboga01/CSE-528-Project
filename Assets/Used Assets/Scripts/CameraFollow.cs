using UnityEngine;

public class CameraFollow : MonoBehaviour
{
    public Transform target;        // Drag your player here
    public float smoothTime = 0.3f; // Delay for smoothness
    public Vector3 offset;          // Adjust to see ahead/above player

    private Vector3 velocity = Vector3.zero;

    void LateUpdate()
    {
        if (target != null)
        {
            // Calculate the desired position
            Vector3 targetPosition = target.position + offset;

            // Keep the camera's original Z-axis so it doesn't vanish
            targetPosition.z = transform.position.z;

            // Smoothly move the camera toward the target
            transform.position = Vector3.SmoothDamp(transform.position, targetPosition, ref velocity, smoothTime);
        }
    }
}