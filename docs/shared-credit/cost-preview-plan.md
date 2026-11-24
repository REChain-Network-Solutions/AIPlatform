# Cost Preview Component - Implementation Plan

## Overview

Reusable Blade + Alpine.js component that shows users how many shared credits an action will cost **before** they execute it. Displayed below the input/form area on each AI feature page.

## Architecture

```
[Frontend Component]          [Backend Endpoint]           [SharedCreditService]
x-cost-preview                POST /cost-preview           resolveUnitCost()
  ├─ watches: generator       ├─ maps generator            getBalance()
  ├─ watches: quantity        │   to EntityEnum             hasBalance()
  └─ fetches on change        ├─ returns JSON
                              └─ web auth (session)
```

## Why New Web Endpoint?

Existing API (`/api/v1/shared-credit/cost/{entity}`) uses `auth:api` (Passport token). Frontend pages use session (web) auth. Need a session-authenticated endpoint.

Also, the current API accepts an entity slug (e.g., `dall-e-3`), but the frontend sends a **generator key** (e.g., `openai`, `stable_diffusion`, `gpt-image-1.5`). The backend needs to map generator → actual EntityEnum model.

## Implementation Steps

### Step 1: Backend Controller

**File:** `app/Http/Controllers/SharedCreditPreviewController.php`

```php
class SharedCreditPreviewController extends Controller
{
    public function preview(Request $request, SharedCreditService $service): JsonResponse
    {
        $request->validate([
            'generator' => 'required|string',
            'quantity'  => 'nullable|integer|min:1|max:10',
        ]);

        $user = $request->user();

        // If user is not on shared credit system, return early
        if (!$user->isSharedCreditUser() || !$service->isEnabled()) {
            return response()->json(['is_shared_credit_user' => false]);
        }

        $entity   = $this->resolveEntity($request->input('generator'));
        $quantity = $request->integer('quantity', 1);
        $unitCost = $service->resolveUnitCost($entity);
        $total    = $unitCost * $quantity;
        $balance  = $service->getBalance($user);

        return response()->json([
            'is_shared_credit_user' => true,
            'model_label'           => $entity->label(),
            'unit_cost'             => $unitCost,
            'quantity'              => $quantity,
            'total_cost'            => $total,
            'balance'               => $balance,
            'sufficient'            => $balance >= $total,
        ]);
    }

    // Maps frontend generator key → EntityEnum
    // Uses same logic as AIController::buildOutput()
    private function resolveEntity(string $generator): EntityEnum
    {
        return match ($generator) {
            'openai'          => EntityEnum::DALL_E_3,    // or DALL_E_2 based on settings
            'stable_diffusion'=> EntityEnum::STABLE_DIFFUSION_XL_1024_V_1_0,
            'gpt-image-1'     => EntityEnum::GPT_IMAGE_1,
            'gpt-image-1.5'   => EntityEnum::GPT_IMAGE_1_5,
            // ... other generators
            default           => EntityEnum::fromSlug($generator),
        };
    }
}
```

### Step 2: Web Route

**File:** `routes/panel.php` (inside user auth group)

```php
Route::post('shared-credit/cost-preview', [SharedCreditPreviewController::class, 'preview'])
    ->name('shared-credit.cost-preview');
```

### Step 3: Blade Component

**File:** `resources/views/default/components/cost-preview.blade.php`

Reusable component that can be dropped into any page:

```blade
{{-- Usage: --}}
{{-- <x-cost-preview generator="generator" quantity="1" /> --}}
{{-- generator and quantity are Alpine.js reactive variable names --}}
```

**Props:**
- `generator` - Alpine.js variable name that holds the current generator/model (string)
- `quantity` - Alpine.js variable name or static number for quantity (default: 1)

**Behavior:**
- Only renders for shared credit users (`$isSharedCreditUser` from server)
- Uses Alpine.js `x-data="costPreview"` component
- Watches generator/quantity changes with debounce (~300ms)
- Shows: icon + "Estimated cost: X credits" or "X credits per image x N = Y total"
- Shows red warning if insufficient balance
- Shows loading state while fetching
- Smooth transition animation

**Visual Design (below form, in the red rectangle area):**
```
┌──────────────────────────────────────────────────────────────────┐
│  ⚡ Stable Diffusion · 1 image · 40 credits    Balance: 500    │
└──────────────────────────────────────────────────────────────────┘
```

Or insufficient:
```
┌──────────────────────────────────────────────────────────────────┐
│  ⚠ Stable Diffusion · 1 image · 40 credits    Balance: 10     │
│    Insufficient credits. Upgrade your plan.                      │
└──────────────────────────────────────────────────────────────────┘
```

### Step 4: Integration in Creative Suite

**File:** `app/Extensions/CreativeSuite/resources/views/home/includes/generator-form.blade.php`

Place the component right after the `</form>` tag (line 225):

```blade
</form>

@if ($isSharedCreditUser)
    <x-cost-preview generator="generator" />
@endif
```

The `generator` Alpine variable already exists in `advancedImageEditorForm` component (line 232), but since `x-cost-preview` is outside the form's `x-data` scope, we need to either:
- **Option A:** Move cost-preview inside the form (simpler, component can access `generator` directly)
- **Option B:** Use Alpine `$dispatch` / `window` events to sync (more decoupled)
- **Option C:** Wrap both form and cost-preview in a parent `x-data` (cleanest)

**Recommended: Option A** - Place inside form, before the closing `</form>` tag.

### Step 5: Tests

**File:** `tests/Feature/SharedCreditCostPreviewTest.php`

Test cases:
- Shared credit user gets cost preview data
- Separated credit user gets `is_shared_credit_user: false`
- Different generators return correct costs
- Quantity multiplies correctly
- DB override costs are reflected
- Insufficient balance flag works
- Guest user is redirected (auth middleware)
- System disabled returns `is_shared_credit_user: false`

## File Changes Summary

| Action | File |
|--------|------|
| **Create** | `app/Http/Controllers/SharedCreditPreviewController.php` |
| **Create** | `resources/views/default/components/cost-preview.blade.php` |
| **Create** | `tests/Feature/SharedCreditCostPreviewTest.php` |
| **Modify** | `routes/panel.php` (add route) |
| **Modify** | `app/Extensions/CreativeSuite/resources/views/home/includes/generator-form.blade.php` (add component) |

## Design Rules

- Use CSS variables / Tailwind design tokens (no hardcoded colors)
- RTL-safe: `ms-*`, `me-*`, `ps-*`, `pe-*`
- Dark mode compatible (uses `hsl(var(--...))`)
- Component is **invisible** for separated credit users (no empty space)
- Debounced fetch to avoid excessive API calls
- `x-transition` for smooth show/hide
- `x-cloak` to prevent FOUC