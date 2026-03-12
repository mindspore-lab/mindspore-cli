// Package train defines domain types for training sessions.
package train

// ProviderKind identifies the infrastructure provider class.
type ProviderKind string

const (
	ProviderLocal  ProviderKind = "local"
	ProviderOnPrem ProviderKind = "onprem"
	ProviderCloud  ProviderKind = "cloud"
)

// BackendKind identifies the execution backend.
type BackendKind string

const (
	BackendLocalProcess BackendKind = "local_process"
	BackendSSHHost      BackendKind = "ssh_host"
	BackendSlurmJob     BackendKind = "slurm_job"
	BackendK8sJob       BackendKind = "k8s_job"
	BackendManagedJob   BackendKind = "managed_job"
)

// TrainTarget describes where training runs.
type TrainTarget struct {
	Provider ProviderKind       `json:"provider"`
	Backend  BackendKind        `json:"backend"`
	Name     string             `json:"name"`
	Config   map[string]any     `json:"config,omitempty"`
}
