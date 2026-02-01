<?php

namespace App\Extensions\ChatSetting\System\Http\Controllers;

use App\Helpers\Classes\Helper;
use App\Http\Controllers\Controller;
use App\Models\ChatCategory;
use Illuminate\Http\Request;

class ChatCategoryController extends Controller
{
    public function index()
    {
        abort_if(setting('chat_setting_for_customer', '1') == '0', 404);

        $list = ChatCategory::query()
            ->where('user_id', auth()->id())
            ->orderBy('name', 'asc')->get();

        return view('chat-setting::category.index', compact('list'));
    }

    public function create()
    {
        abort_if(setting('chat_setting_for_customer', '1') == '0', 404);

        return view('chat-setting::category.form', [
            'method' => 'post',
            'title'  => 'Create Chat Category',
            'action' => route('dashboard.user.chat-setting.chat-category.store'),
            'item'   => new ChatCategory,
        ]);
    }

    public function store(Request $request): \Illuminate\Http\RedirectResponse
    {
        $request->validate([
            'name' => 'required|string|max:255',
        ]);

        ChatCategory::query()->create([
            'name'    => $request->input('name'),
            'user_id' => auth()->id(),
        ]);

        return redirect()->route('dashboard.user.chat-setting.chat-category.index')->with([
            'message' => 'Chat Category has been created',
            'type'    => 'success',
        ]);
    }

    public function edit(ChatCategory $chatCategory)
    {
        if ($chatCategory->user_id != auth()->id()) {
            abort(403, 'Unauthorized');
        }

        return view('chat-setting::category.form', [
            'method' => 'put',
            'title'  => 'Edit Chat Category',
            'action' => route('dashboard.user.chat-setting.chat-category.update', $chatCategory),
            'item'   => $chatCategory,
        ]);
    }

    public function update(Request $request, ChatCategory $chatCategory)
    {
        $request->validate([
            'name' => 'required|string|max:255',
        ]);

        $chatCategory->update([
            'name' => $request->input('name'),
        ]);

        return redirect()
            ->route('dashboard.user.chat-setting.chat-category.index')->with([
                'message' => 'Chat Category has been updated',
                'type'    => 'success',
            ]);
    }

    public function destroy(ChatCategory $chatCategory): \Illuminate\Http\JsonResponse
    {
        if (Helper::appIsDemo()) {
            return response()->json([
                'status'  => 'error',
                'message' => trans('This feature is disabled in demo mode.'),
            ]);
        }

        $chatCategory->delete();

        return response()->json([
            'message'    => trans('Category Deleted Successfully'),
            'reload'     => true,
            'setTimeOut' => 1000,
        ]);
    }
}
