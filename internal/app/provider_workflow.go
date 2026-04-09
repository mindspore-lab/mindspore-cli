package app

import (
	"fmt"
	"sort"
	"strings"

	"github.com/mindspore-lab/mindspore-cli/ui/model"
)

type providerWorkflowState struct {
	catalog            *providerCatalog
	authState          *providerAuthState
	effectiveAuthState *providerAuthState
	importSuggestions  []providerImportSuggestion
	modelState         *modelSelectionState
	loggedIn           bool
}

type logicalModelSelectionResult struct {
	ProviderLabel string
}

type providerCatalogLoadMode int

const (
	providerCatalogLoadCacheFirst providerCatalogLoadMode = iota
	providerCatalogLoadBlocking
)

func (a *Application) loadProviderWorkflowState(mode providerCatalogLoadMode) (*providerWorkflowState, error) {
	appCfg, err := loadAppConfig()
	if err != nil {
		return nil, err
	}
	var catalog *providerCatalog
	switch mode {
	case providerCatalogLoadBlocking:
		catalog, err = loadProviderCatalogBlocking(appCfg.ExtraProviders)
	default:
		catalog, err = loadProviderCatalog(nil, appCfg.ExtraProviders)
	}
	if err != nil {
		return nil, err
	}
	authState, err := loadProviderAuthState()
	if err != nil {
		return nil, err
	}
	modelState, err := loadModelSelectionState()
	if err != nil {
		return nil, err
	}
	state := &providerWorkflowState{
		catalog:    catalog,
		authState:  authState,
		modelState: modelState,
		loggedIn:   isLoggedIn(),
	}
	state.refreshDerivedProviderState()
	return state, nil
}

func (s *providerWorkflowState) refreshDerivedProviderState() {
	if s == nil {
		return
	}
	s.importSuggestions = detectProviderImportSuggestions(s.catalog, s.authState)
	s.effectiveAuthState = cloneProviderAuthState(s.authState)
	if s.effectiveAuthState == nil {
		s.effectiveAuthState = emptyProviderAuthState()
	}
}

func (a *Application) emitConnectPopup(canEscape bool) {
	state, err := a.loadProviderWorkflowState(providerCatalogLoadCacheFirst)
	if err != nil {
		a.emitToolError("connect", "Failed to load provider catalog: %v", err)
		return
	}

	options := buildConnectProviderOptions(state.catalog, state.loggedIn)

	a.EventCh <- model.Event{
		Type: model.ModelSetupOpen,
		SetupPopup: &model.SetupPopup{
			Title:         "Connect Provider",
			InputLabel:    "API key",
			Screen:        model.SetupScreenPresetPicker,
			PresetOptions: options,
			PresetSelected: (&model.SetupPopup{
				PresetOptions: options,
			}).FirstSelectablePreset(),
			CanEscape:  canEscape,
			BackCloses: true,
		},
	}
}

func (a *Application) emitModelBrowser() {
	state, err := a.loadProviderWorkflowState(providerCatalogLoadCacheFirst)
	if err != nil {
		a.emitToolError("model", "Failed to load models: %v", err)
		return
	}
	a.emitModelBrowserWithState(state, "")
}

func (a *Application) emitModelBrowserWithState(state *providerWorkflowState, preferredProviderID string) {
	if state == nil {
		a.emitToolError("model", "Failed to load model browser state")
		return
	}

	a.EventCh <- model.Event{
		Type:         model.ModelBrowserOpen,
		ModelBrowser: buildModelBrowserPopup(state, preferredProviderID),
	}
}

func (a *Application) emitModelPicker() {
	state, err := a.loadProviderWorkflowState(providerCatalogLoadCacheFirst)
	if err != nil {
		a.emitToolError("model", "Failed to load models: %v", err)
		return
	}
	a.emitModelPickerWithState(state)
}

func (a *Application) emitModelPickerWithState(state *providerWorkflowState) {
	if state == nil {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "No models available. Run /model to configure a provider.",
		}
		return
	}

	options := buildModelPickerOptions(state.catalog, state.effectiveAuthState, state.modelState, state.loggedIn, state.importSuggestions)
	if len(options) == 0 {
		a.EventCh <- model.Event{
			Type:    model.AgentReply,
			Message: "No models available. Run /model to configure a provider.",
		}
		return
	}

	selected := 0
	if state.modelState != nil && state.modelState.Active != nil {
		activeID := state.modelState.Active.key()
		for i, opt := range options {
			if opt.ID == activeID {
				selected = i
				break
			}
		}
	} else {
		selected = firstSelectableOptionIndex(options)
	}

	a.EventCh <- model.Event{
		Type: model.ModelPickerOpen,
		Popup: &model.SelectionPopup{
			Title:       "Select model",
			Options:     options,
			Selected:    selected,
			ActionID:    "model_picker",
			SearchQuery: "",
		},
	}
}

func buildModelBrowserPopup(state *providerWorkflowState, preferredProviderID string) *model.ModelBrowserPopup {
	providerOptions := buildConnectProviderOptions(state.catalog, state.loggedIn)
	modelOptions := buildModelPickerOptions(state.catalog, state.effectiveAuthState, state.modelState, state.loggedIn, state.importSuggestions)

	providerSelected := firstSelectableOptionIndex(providerOptions)
	if activeProviderID := activeModelProviderID(state.modelState); activeProviderID != "" {
		if idx := optionIndexByID(providerOptions, activeProviderID); idx >= 0 {
			providerSelected = idx
		}
	}
	if preferredProviderID = normalizedProviderID(preferredProviderID); preferredProviderID != "" {
		if idx := optionIndexByID(providerOptions, preferredProviderID); idx >= 0 {
			providerSelected = idx
		}
	}

	modelSelected := firstSelectableModelIndex(modelOptions)
	if state.modelState != nil && state.modelState.Active != nil {
		if idx := optionIndexByID(modelOptions, state.modelState.Active.key()); idx >= 0 {
			modelSelected = idx
		}
	}
	if preferredProviderID != "" {
		for i, opt := range modelOptions {
			if strings.HasPrefix(opt.ID, preferredProviderID+":") && opt.Selectable() {
				modelSelected = i
				break
			}
		}
	}

	popup := &model.ModelBrowserPopup{
		Providers: model.SelectionPopup{
			Title:    "Providers",
			Options:  providerOptions,
			Selected: providerSelected,
		},
		Models: model.SelectionPopup{
			Title:    "Models",
			Options:  modelOptions,
			Selected: modelSelected,
		},
		Focus:            model.ModelBrowserFocusModel,
		ProvidersVisible: false,
	}
	if len(modelOptions) == 0 {
		popup.Focus = model.ModelBrowserFocusProvider
		popup.ProvidersVisible = true
	}
	return popup
}

func buildConnectProviderOptions(catalog *providerCatalog, loggedIn bool) []model.SelectionOption {
	if catalog == nil {
		return nil
	}
	popularProviders, otherProviders := partitionConnectProviders(catalog.Providers, loggedIn)
	options := make([]model.SelectionOption, 0, len(catalog.Providers)+4)
	if len(popularProviders) > 0 {
		options = append(options, model.SelectionOption{ID: "__header__popular", Label: "Popular", Header: true, Disabled: true})
	}
	for _, provider := range popularProviders {
		options = append(options, connectProviderOption(provider, loggedIn))
	}
	if len(otherProviders) > 0 {
		if len(options) > 0 {
			options = append(options, model.SelectionOption{ID: "__separator__connect", Separator: true, Disabled: true})
		}
		options = append(options, model.SelectionOption{ID: "__header__other", Label: "Other", Header: true, Disabled: true})
	}
	for _, provider := range otherProviders {
		options = append(options, connectProviderOption(provider, loggedIn))
	}
	return options
}

func connectProviderOption(provider providerCatalogEntry, loggedIn bool) model.SelectionOption {
	opt := model.SelectionOption{
		ID:            provider.ID,
		Label:         provider.Label,
		RequiresInput: provider.Protocol != "mindspore-cli-free",
	}
	if provider.Protocol == "mindspore-cli-free" {
		opt.Desc = "(Recommended)"
		opt.RequiresInput = false
		return opt
	}
	return opt
}

func buildProviderScopedModelPicker(provider providerCatalogEntry) *model.SelectionPopup {
	options := make([]model.SelectionOption, 0, len(provider.Models))
	for _, m := range provider.Models {
		options = append(options, model.SelectionOption{
			ID:    modelRef{ProviderID: provider.ID, ModelID: m.ID}.key(),
			Label: m.Label,
		})
	}
	return &model.SelectionPopup{
		Title:       "Select model",
		Options:     options,
		Selected:    firstSelectableOptionIndex(options),
		ActionID:    "connect_provider_model_picker:" + provider.ID,
		SearchQuery: "",
	}
}

func firstSelectableOptionIndex(options []model.SelectionOption) int {
	for i, opt := range options {
		if opt.Selectable() {
			return i
		}
	}
	return 0
}

func optionIndexByID(options []model.SelectionOption, id string) int {
	needle := strings.TrimSpace(id)
	for i, opt := range options {
		if opt.ID == needle {
			return i
		}
	}
	return -1
}

func firstSelectableModelIndex(options []model.SelectionOption) int {
	for i, opt := range options {
		if opt.Selectable() && !opt.ProviderRow {
			return i
		}
	}
	return firstSelectableOptionIndex(options)
}

func activeModelProviderID(state *modelSelectionState) string {
	if state == nil || state.Active == nil {
		return ""
	}
	return normalizedProviderID(state.Active.ProviderID)
}

func buildModelPickerOptions(catalog *providerCatalog, authState *providerAuthState, modelState *modelSelectionState, loggedIn bool, importSuggestions []providerImportSuggestion) []model.SelectionOption {
	if catalog == nil {
		return nil
	}
	usable := usableProviders(catalog, authState, loggedIn)
	importSuggestionsByID := providerImportSuggestionByID(importSuggestions)
	options := make([]model.SelectionOption, 0)
	const recentDisplayLimit = 5

	addModelRefs := func(title string, refs []modelRef, limit int, showProvider bool) {
		if len(refs) == 0 {
			return
		}

		group := make([]model.SelectionOption, 0, len(refs)+1)
		group = append(group, model.SelectionOption{ID: "__header__" + title, Label: title, Header: true, Disabled: true})
		count := 0
		for _, ref := range refs {
			provider, ok := findProviderInList(usable, ref.ProviderID)
			if !ok {
				continue
			}
			label := modelLabelForRef(provider, ref.ModelID)
			desc := ""
			if showProvider {
				desc = "· " + provider.Label
			}
			group = append(group, model.SelectionOption{
				ID:    ref.key(),
				Label: label,
				Desc:  desc,
			})
			count++
			if limit > 0 && count >= limit {
				break
			}
		}
		if count == 0 {
			return
		}
		if len(options) > 0 {
			options = append(options, model.SelectionOption{ID: "__separator__" + title, Separator: true, Disabled: true})
		}
		options = append(options, group...)
	}

	if modelState != nil {
		addModelRefs("Recent", modelState.Recents, recentDisplayLimit, true)
		addModelRefs("Favorites", modelState.Favorites, 0, false)
	}

	for _, provider := range usable {
		if len(options) > 0 {
			options = append(options, model.SelectionOption{ID: "__separator__provider:" + provider.ID, Separator: true, Disabled: true})
		}
		providerRow := model.SelectionOption{
			ID:          "__provider__" + provider.ID,
			Label:       provider.Label,
			ProviderRow: true,
		}
		if suggestion, ok := importSuggestionsByID[provider.ID]; ok {
			providerRow.Desc = suggestion.SourceLabel
		}
		if provider.Protocol == "mindspore-cli-free" {
			providerRow.Disabled = true
		} else if importSuggestionsByID[provider.ID].ProviderID == "" {
			providerRow.DeleteProviderID = provider.ID
		}
		options = append(options, providerRow)
		for _, m := range provider.Models {
			options = append(options, model.SelectionOption{
				ID:    modelRef{ProviderID: provider.ID, ModelID: m.ID}.key(),
				Label: m.Label,
			})
		}
	}
	return options
}

type deleteProviderResult struct {
	Fallback *modelRef
	Cleared  bool
	State    *providerWorkflowState
}

func (a *Application) deleteConnectedProvider(providerID string) (deleteProviderResult, error) {
	state, err := a.loadProviderWorkflowState(providerCatalogLoadCacheFirst)
	if err != nil {
		return deleteProviderResult{}, err
	}
	providerID = normalizedProviderID(providerID)
	if providerID == "" {
		return deleteProviderResult{}, fmt.Errorf("provider id is empty")
	}
	if state.authState == nil {
		state.authState = emptyProviderAuthState()
	}
	delete(state.authState.Providers, providerID)
	if err := saveProviderAuthState(state.authState); err != nil {
		return deleteProviderResult{}, err
	}
	state.refreshDerivedProviderState()

	if state.modelState == nil {
		state.modelState = emptyModelSelectionState()
	}
	wasActive := state.modelState.Active != nil && normalizedProviderID(state.modelState.Active.ProviderID) == providerID
	state.modelState.Active = removeActiveProviderRef(state.modelState.Active, providerID)
	state.modelState.Recents = removeProviderRefs(state.modelState.Recents, providerID)
	state.modelState.Favorites = removeProviderRefs(state.modelState.Favorites, providerID)
	if err := saveModelSelectionState(state.modelState); err != nil {
		return deleteProviderResult{}, err
	}

	result := deleteProviderResult{State: state}
	if !wasActive {
		return result, nil
	}

	fallback := firstAvailableModelRef(state.catalog, state.effectiveAuthState, state.modelState, state.loggedIn)
	if fallback == nil {
		if blockingState, blockingErr := a.loadProviderWorkflowState(providerCatalogLoadBlocking); blockingErr == nil {
			state = blockingState
			result.State = blockingState
			fallback = firstAvailableModelRef(state.catalog, state.effectiveAuthState, state.modelState, state.loggedIn)
		}
	}
	if fallback == nil {
		if err := a.clearActiveLogicalModel(); err != nil {
			return deleteProviderResult{}, err
		}
		result.Cleared = true
		return result, nil
	}
	if _, err := a.activateLogicalModelSelection(fallback.ProviderID, fallback.ModelID); err != nil {
		return deleteProviderResult{}, err
	}
	result.Fallback = fallback
	return result, nil
}

func removeActiveProviderRef(ref *modelRef, providerID string) *modelRef {
	if ref == nil {
		return nil
	}
	if normalizedProviderID(ref.ProviderID) == normalizedProviderID(providerID) {
		return nil
	}
	normalized := ref.normalized()
	return &normalized
}

func removeProviderRefs(refs []modelRef, providerID string) []modelRef {
	out := make([]modelRef, 0, len(refs))
	for _, ref := range refs {
		if normalizedProviderID(ref.ProviderID) == normalizedProviderID(providerID) {
			continue
		}
		out = append(out, ref)
	}
	return out
}

func firstAvailableModelRef(catalog *providerCatalog, authState *providerAuthState, modelState *modelSelectionState, loggedIn bool) *modelRef {
	usable := usableProviders(catalog, authState, loggedIn)
	tryRefs := func(refs []modelRef) *modelRef {
		for _, ref := range refs {
			provider, ok := findProviderInList(usable, ref.ProviderID)
			if !ok {
				continue
			}
			for _, m := range provider.Models {
				if m.ID == strings.TrimSpace(ref.ModelID) {
					normalized := ref.normalized()
					return &normalized
				}
			}
		}
		return nil
	}
	if modelState != nil {
		if ref := tryRefs(modelState.Recents); ref != nil {
			return ref
		}
		if ref := tryRefs(modelState.Favorites); ref != nil {
			return ref
		}
	}
	for _, provider := range usable {
		if len(provider.Models) == 0 {
			continue
		}
		ref := modelRef{ProviderID: provider.ID, ModelID: provider.Models[0].ID}.normalized()
		return &ref
	}
	return nil
}

func partitionConnectProviders(providers []providerCatalogEntry, loggedIn bool) ([]providerCatalogEntry, []providerCatalogEntry) {
	popular := make([]providerCatalogEntry, 0, 5)
	other := make([]providerCatalogEntry, 0, len(providers))

	providerByID := make(map[string]providerCatalogEntry, len(providers))
	for _, provider := range providers {
		providerByID[provider.ID] = provider
	}

	if loggedIn {
		if provider, ok := providerByID[mindsporeCLIFreeProviderID]; ok {
			popular = append(popular, provider)
			delete(providerByID, mindsporeCLIFreeProviderID)
		}
	}

	for _, id := range connectPopularProviderIDs {
		if provider, ok := providerByID[id]; ok {
			popular = append(popular, provider)
			delete(providerByID, id)
		}
	}

	if !loggedIn {
		delete(providerByID, mindsporeCLIFreeProviderID)
	}

	rest := make([]providerCatalogEntry, 0, len(providerByID))
	for _, provider := range providerByID {
		rest = append(rest, provider)
	}
	sort.Slice(rest, func(i, j int) bool {
		return strings.ToLower(rest[i].Label) < strings.ToLower(rest[j].Label)
	})
	other = append(other, rest...)
	return popular, other
}

func usableProviders(catalog *providerCatalog, authState *providerAuthState, loggedIn bool) []providerCatalogEntry {
	if catalog == nil {
		return nil
	}
	out := make([]providerCatalogEntry, 0, len(catalog.Providers))
	for _, provider := range catalog.Providers {
		if provider.Protocol == "mindspore-cli-free" {
			if loggedIn {
				out = append(out, provider)
			}
			continue
		}
		if authState != nil {
			if entry, ok := authState.Providers[provider.ID]; ok && strings.TrimSpace(entry.APIKey) != "" {
				out = append(out, provider)
			}
		}
	}
	return out
}

func findProviderInList(providers []providerCatalogEntry, id string) (providerCatalogEntry, bool) {
	for _, provider := range providers {
		if provider.ID == normalizedProviderID(id) {
			return provider, true
		}
	}
	return providerCatalogEntry{}, false
}

func modelLabelForRef(provider providerCatalogEntry, modelID string) string {
	for _, m := range provider.Models {
		if m.ID == strings.TrimSpace(modelID) {
			return m.Label
		}
	}
	return strings.TrimSpace(modelID)
}

func providerDisplayLabel(catalog *providerCatalog, providerID string) string {
	if catalog != nil {
		if provider, ok := catalog.Provider(providerID); ok && strings.TrimSpace(provider.Label) != "" {
			return strings.TrimSpace(provider.Label)
		}
	}
	return runtimeProviderDisplayLabel(providerID)
}

func runtimeProviderDisplayLabel(providerID string) string {
	switch normalizedProviderID(providerID) {
	case "openai", "openai-completion":
		return "OpenAI"
	case "openai-responses":
		return "OpenAI Responses"
	case "anthropic":
		return "Anthropic"
	case mindsporeCLIFreeProviderID:
		return "MindSpore CLI Free"
	default:
		if label := canonicalProviderLabel(providerID, ""); label != "" {
			return label
		}
		return strings.TrimSpace(providerID)
	}
}

func (a *Application) connectProvider(providerID, apiKey string) (*providerWorkflowState, error) {
	state, err := a.loadProviderWorkflowState(providerCatalogLoadCacheFirst)
	if err != nil {
		return nil, err
	}
	provider, ok := state.catalog.Provider(providerID)
	if !ok {
		state, err = a.loadProviderWorkflowState(providerCatalogLoadBlocking)
		if err != nil {
			return nil, err
		}
		provider, ok = state.catalog.Provider(providerID)
	}
	if !ok {
		return nil, fmt.Errorf("unknown provider %q", providerID)
	}

	if provider.Protocol == "mindspore-cli-free" {
		if !state.loggedIn {
			return nil, fmt.Errorf("MindSpore CLI Free requires /login first")
		}
		return state, nil
	}

	apiKey = strings.TrimSpace(apiKey)
	if apiKey == "" {
		return nil, fmt.Errorf("provider %s requires api key", provider.Label)
	}
	state.authState.Providers[provider.ID] = providerAuthEntry{
		ProviderID: provider.ID,
		APIKey:     apiKey,
	}
	if err := saveProviderAuthState(state.authState); err != nil {
		return nil, err
	}
	state.refreshDerivedProviderState()
	return state, nil
}

func (a *Application) activateLogicalModelSelection(providerID, modelID string) (logicalModelSelectionResult, error) {
	state, err := a.loadProviderWorkflowState(providerCatalogLoadCacheFirst)
	if err != nil {
		return logicalModelSelectionResult{}, err
	}
	if _, ok := state.catalog.Provider(providerID); !ok {
		state, err = a.loadProviderWorkflowState(providerCatalogLoadBlocking)
		if err != nil {
			return logicalModelSelectionResult{}, err
		}
	}

	resolved, presetID, err := resolveRuntimeSelection(state.catalog, state.effectiveAuthState, modelRef{
		ProviderID: providerID,
		ModelID:    modelID,
	})
	if err != nil {
		return logicalModelSelectionResult{}, err
	}

	a.restoreModelConfigFromPreset()
	a.Config.Model.URL = resolved.URL
	if err := a.SetProvider(resolved.Provider, resolved.Model, resolved.Key); err != nil {
		return logicalModelSelectionResult{}, err
	}

	a.activeModelPresetID = presetID
	if a.activeModelPresetID == "" {
		a.activeModelPresetID = ""
	}

	if state.modelState == nil {
		state.modelState = emptyModelSelectionState()
	}
	selected := modelRef{ProviderID: providerID, ModelID: modelID}.normalized()
	state.modelState.Active = &selected
	state.modelState.Recents = append([]modelRef{selected}, state.modelState.Recents...)
	state.modelState.normalize()
	if err := saveModelSelectionState(state.modelState); err != nil {
		return logicalModelSelectionResult{}, err
	}
	return logicalModelSelectionResult{
		ProviderLabel: providerDisplayLabel(state.catalog, providerID),
	}, nil
}
