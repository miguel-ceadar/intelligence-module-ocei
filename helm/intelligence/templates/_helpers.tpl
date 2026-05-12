{{/* Standard chart helpers. */}}

{{- define "intelligence.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "intelligence.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "intelligence.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "intelligence.labels" -}}
helm.sh/chart: {{ include "intelligence.chart" . }}
{{ include "intelligence.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "intelligence.selectorLabels" -}}
app.kubernetes.io/name: {{ include "intelligence.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/* Secret name resolution — prefer existingSecretName if set. */}}
{{- define "intelligence.secretName" -}}
{{- if .Values.existingSecretName -}}
{{ .Values.existingSecretName }}
{{- else -}}
{{ include "intelligence.fullname" . }}-secrets
{{- end -}}
{{- end -}}
