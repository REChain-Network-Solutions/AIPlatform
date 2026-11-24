<?php

namespace App\Http\Controllers\Dashboard;

use App\Http\Controllers\Controller;
use App\Http\Requests\UpdateChannelRequest;
use App\Services\UpdateChannel;
use Illuminate\Http\JsonResponse;
use Illuminate\Support\Facades\Log;
use Throwable;

class UpdateChannelController extends Controller
{
    public function update(UpdateChannelRequest $request): JsonResponse
    {
        abort_unless($request->user()?->isAdmin(), 403);

        try {
            $channel = UpdateChannel::set(
                $request->boolean('enable_beta') ? UpdateChannel::BETA : UpdateChannel::STABLE
            );

            Log::info('Update channel changed', [
                'user_id' => $request->user()->id,
                'channel' => $channel,
            ]);

            return response()->json([
                'success' => true,
                'message' => $channel === UpdateChannel::BETA
                    ? __('Successfully switched to Beta update channel. Page will reload to apply changes.')
                    : __('Successfully switched to Stable update channel. Page will reload to apply changes.'),
                'channel' => $channel,
                'reload'  => true,
            ]);
        } catch (Throwable $th) {
            Log::error('Update channel change failed', [
                'user_id' => $request->user()?->id,
                'error'   => $th->getMessage(),
            ]);

            return response()->json([
                'success' => false,
                'message' => __('An unexpected error occurred. Please try again or contact support.'),
            ], 500);
        }
    }
}
