# Cost Preview - Added Pages

The cost preview component (`<x-cost-preview />`) has been added to the following pages.

## API Route

| Method | Route | Controller | Name |
|--------|-------|------------|------|
| POST | `shared-credit/cost-preview` | `SharedCreditPreviewController@preview` | `shared-credit.cost-preview` |

## Added Pages

### 1. Creative Suite
- **URL:** `dashboard/user/creative-suite`
- **Location:** Right below the generator form
- **Model source:** `generator` select dropdown (`openai`, `stable_diffusion`, `gpt-image-1`, etc.)
- **Quantity:** Fixed 1

### 2. AI Chat Image
- **URL:** `ai-chat-image/chat`
- **Location:** Right above the prompt form
- **Model source:** Top-left model dropdown (`$store.chatsV2.selectedModel`)
- **Quantity:** `image_count` select (1-4)

### 3. AI Image Pro - Realtime
- **URL:** `dashboard/user/ai-image-pro/realtime`
- **Location:** Right above the prompt bar
- **Model source:** Fixed (`BLACK_FOREST_LABS_FLUX_1_SCHNELL`)
- **Quantity:** Fixed 1

### 4. AI Realtime Image
- **URL:** `dashboard/user/ai-realtime-image`
- **Location:** Right below the generator form
- **Model source:** Fixed (`BLACK_FOREST_LABS_FLUX_1_SCHNELL`)
- **Quantity:** Fixed 1

### 5. AI Image Generator (Legacy)
- **URL:** `dashboard/user/openai/generator/ai_image_generator`
- **Location:** Right side of the Advanced Settings button (in each model tab)
- **Model source:** Model tabs (`openai`, `gpt_image_1`, `gpt_image_1_5`, `stable_diffusion`, etc.)
- **Quantity:** `#image_number_of_images` select (1-5)
- **Note:** For shared credit users, the legacy credit bar (`<x-credit-list>`) is hidden
- **Extension tabs:** Midjourney, Flux Pro, Flux 2 Flex, Ideogram, Nano Banana, Nano Banana Pro, Nano Banana 2

### 6. Photo Studio
- **URL:** `dashboard/user/photo-studio`
- **Location:** Below the Generate button
- **Model source:** Fixed - from admin settings (`setting('default_photo_studio')` -> `clipdrop` or `novita`)
- **Quantity:** Fixed 1
- **Note:** Credit is recalculated when the action changes (`action` parameter is sent to the backend)

### 7. AI Product Shot
- **URL:** `dashboard/user/ai-product-shot`
- **Location:** Below the Generate button
- **Model source:** Fixed (`pebblely`)
- **Quantity:** Fixed 1

### 8. AI Video to Video
- **URL:** `dashboard/user/openai/generator/ai_video_to_video`
- **Location:** Below the Generate button
- **Model source:** `#model` select dropdown (`video-upscaler`, `cogvideox-5b/video-to-video`, `animatediff-v2v`, `fast-animatediff/turbo/video-to-video`)
- **Quantity:** Fixed 1
- **Note:** Credit is recalculated when the model changes

### 9. AI Video Pro
- **URL:** `dashboard/user/ai-video-pro`
- **Location:** Below the Generate button
- **Model source:** `selectedFeature` (Alpine.js) - 3-layer selection: Action -> Sub Model -> Feature
- **Entity:** `EntityEnum::fromSlug(feature)` - Sora 2, Veo 2, Kling, Luma, Minimax, Grok, etc.
- **Quantity:** Fixed 1
- **Note:** Credit is automatically recalculated via `x-effect` when the feature (model) changes

### 10. AI Persona
- **URL:** `dashboard/user/ai-persona/create`
- **Location:** Below the Generate Video button
- **Model source:** Fixed (`heygen`)
- **Quantity:** Fixed 1

### 11. AI Video (Image to Video)
- **URL:** `dashboard/user/openai/generator/ai_video`
- **Location:** Below the Generate button
- **Model source:** Fixed (`image-to-video`)
- **Quantity:** Fixed 1

### 12. AI Avatar
- **URL:** `dashboard/user/ai-avatar/create`
- **Location:** Below the Generate Video button
- **Model source:** Fixed (`synthesia`)
- **Quantity:** Fixed 1

### 13. Advanced Image - Editor
- **URL:** `dashboard/user/advanced-image` (editor mode)
- **Location:** Below the Generate button in the editor sidebar
- **Model source:** `aiModel` Alpine variable (clipdrop, novita, freepik, gpt-image-1, flux-pro/kontext, nano-banana/edit, etc.)
- **Action:** `selectedTool` Alpine variable (merge_face, cleanup, reimagine, remove_background, etc.)
- **Quantity:** Fixed 1
- **Note:** Credit is recalculated via `x-effect` when either model or tool changes. For `flux-pro/kontext` + certain tools (`cleanup`, `style_transfer`, `image_relight`), a different entity (`FLUX_PRO_KONTEXT_MAX_MULTI`) is used.

### 14. Advanced Image - Home *(newly added)*
- **URL:** `dashboard/user/advanced-image`
- **Location:** Right below the generator form (below the submit button)
- **Model source:** `generator` Alpine variable (`stable_diffusion`, `openai`, `flux-pro`, `gpt-image-1`, `gpt-image-1-5`, `flux-pro-kontext`, `flux-2-flex`, `ideogram`, `nano-banana`, etc.)
- **Quantity:** Fixed 1
- **Note:** Event is dispatched via `init()` + `$nextTick`, credit is recalculated via `$watch` when the generator changes.

### 15. AI Image Pro *(newly added)*
- **URL:** `dashboard/user/ai-image-pro`
- **Location:** Right below the generator form (below the sticky form)
- **Model source:** `currentModel.slug` Alpine variable - `aiImageProGeneratorForm` component, selected from a dynamic model list (`$activeImageModels`)
- **Quantity:** Fixed 1
- **Note:** Event is dispatched via `init()` + `$nextTick`, credit is recalculated via `$watch` when `selectedModel` changes. Model preference is stored in localStorage.

### 16. AI Music *(newly added)*
- **URL:** `dashboard/user/ai-music/create`
- **Location:** Below the Generate Song button
- **Model source:** Fixed - from admin settings (`Setting::getCache()->ai_music_model` -> `music-01` default)
- **Quantity:** Fixed 1
- **Note:** Event is dispatched via `x-init` + `$nextTick` on page load. Model does not change (set by admin).

### 17. AI Voiceover *(updated)*
- **URL:** `dashboard/user/openai/generator/ai_voiceover`
- **Location:** Below the Generate button
- **Model source:** Based on admin TTS setting (`$settings_two->tts`): `tts-1` (OpenAI), `google`, `elevenlabs`/`eleven_v3`, `azure`, `speechify`
- **Quantity:** Dynamic — speech count (number of `.speech` elements in `.speeches` container, minimum 1)
- **Note:** Uses `MutationObserver` on `.speeches` container to reactively update cost when speeches are added/removed. Each speech costs 5.0 credits.

### 18. AI Speech to Text *(newly added)*
- **URL:** `dashboard/user/openai/generator/ai_speech_to_text`
- **Location:** Below the Generate button (inside the generator_others component)
- **Model source:** Fixed (`whisper-1`)
- **Quantity:** Fixed 1

### 19. AI Voice Isolator *(newly added)*
- **URL:** `dashboard/user/openai/generator/ai_voice_isolator`
- **Location:** Below the Isolate Voice button
- **Model source:** Fixed (`isolator`)
- **Quantity:** Fixed 1

### 20. AI Music Pro *(updated)*
- **URL:** `dashboard/user/ai-music-pro`
- **Location:** Below the Generate button
- **Model source:** Fixed (`elevenlabs-ai-music`)
- **Quantity:** Dynamic — `duration / 60` (30s=0.5, 1min=1, 2min=2, 3min=3, None=1 default)
- **Note:** Listens to `#duration` dropdown changes and recalculates cost. Backend formula: `(duration_seconds / 60) × 10.0` credits.

### 21. Fashion Studio Photoshoots *(newly added)*
- **URL:** `dashboard/user/fashion-studio/photo_shoots`
- **Location:** Below the Generate button, inside the left panel card
- **Model source:** Fixed (`nano-banana-pro/edit`)
- **Quantity:** `numImages` from user settings (`FashionStudioUserSetting.num_images`, default 2, max 4)
- **Note:** Quantity is determined server-side from user settings and passed to the view. Each image costs 80.0 credits.

### 22. Fashion Studio Virtual Try-On *(newly added)*
- **URL:** `dashboard/user/fashion-studio/virtual_try_on`
- **Location:** Below the Generate button, inside the left panel
- **Model source:** Fixed (`nano-banana-pro/edit`)
- **Quantity:** Fixed 1
- **Note:** Virtual try-on always generates 1 image (80.0 credits).

### 23. Fashion Studio Change Model *(newly added)*
- **URL:** `dashboard/user/fashion-studio/change_model`
- **Location:** Below the Generate button, inside the left panel
- **Model source:** Fixed (`nano-banana-pro/edit`)
- **Quantity:** Fixed 1
- **Note:** Change model always generates 1 image (80.0 credits).

### 24. Fashion Studio Edit Image *(newly added)*
- **URL:** `dashboard/user/fashion-studio/edit_image`
- **Location:** Below the Generate button, inside the left panel
- **Model source:** Fixed (`nano-banana-pro/edit`)
- **Quantity:** Fixed 1
- **Note:** Edit image always generates 1 image (80.0 credits).

### 25. Fashion Studio Create Video *(newly added)*
- **URL:** `dashboard/user/fashion-studio/create_video`
- **Location:** Below the Generate Video button, inside the left panel
- **Model source:** Dynamic — from admin setting (`fashion-studio-video-default-model`, default `veo-3-1-image-to-video`)
- **Quantity:** Fixed 1
- **Note:** Video entity is resolved server-side via `getVideoModelEntity()` and passed to the view.
