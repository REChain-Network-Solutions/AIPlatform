<?php

namespace App\Extensions\AISocialMedia\System\Http\Controllers;

use App\Extensions\AISocialMedia\System\Enums\Platform;
use App\Extensions\AISocialMedia\System\Models\AutomationPlatform;
use App\Http\Controllers\Controller;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;

class AutomationPlatformController extends Controller
{
    public function index()
    {
        $platforms = AutomationPlatform::query()
            ->where('user_id', Auth::id())->get();

        return view('ai-social-media::platforms.list', [
            'platformX' => $platforms
                ->firstWhere('platform', Platform::x),
            'platformLinkedin' => $platforms
                ->where('platform', Platform::linkedin)
                ->first(),
            'platformInstagram' => $platforms
                ->where('platform', Platform::instagram)
                ->first(),
        ]);
    }

    public function update(Request $request, Platform $platform): RedirectResponse
    {
        $data = $request->validate($this->validatePlatform($platform));

        AutomationPlatform::query()
            ->updateOrCreate([
                'user_id'  => Auth::id(),
                'platform' => $platform->value,
            ], [
                'credentials'  => $data,
                'connected_at' => now(),
                'expires_at'   => now()->addDays(30),
            ]);

        return back()->with([
            'type'    => 'success',
            'message' => 'Platform updated successfully',
        ]);
    }

    public function disconnect(AutomationPlatform $automationPlatform): RedirectResponse
    {
        $automationPlatform->delete();

        return back()->with([
            'type'    => 'success',
            'message' => 'Platform disconnected successfully',
        ]);
    }

    public function validatePlatform(Platform $platform): array
    {

        return match ($platform) {
            Platform::x => [
                'account_id'          => 'required',
                'access_token_secret' => 'required',
                'access_token'        => 'required',
                'bearer_token'        => 'required',
                'consumer_secret'     => 'required',
                'consumer_key'        => 'required',
            ],
            Platform::linkedin => [
                'access_token' => 'required',
            ],
            default => [],
        };
    }
}
