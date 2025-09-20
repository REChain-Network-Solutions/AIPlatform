<?php

namespace App\Http\Controllers\Admin\Frontend;

use App\Http\Controllers\Controller;
use App\Models\Frontend\Curtain;
use Illuminate\Http\RedirectResponse;
use Illuminate\Http\Request;
use Illuminate\Http\UploadedFile;

class CurtainController extends Controller
{
    public function index()
    {
        $items = Curtain::query()->get();

        return view('default.panel.admin.frontend.curtain.index', compact('items'));
    }

    public function edit(Curtain $curtain)
    {
        return view('default.panel.admin.frontend.curtain.form', [
            'item' => $curtain,
        ]);
    }

    public function update(Request $request, Curtain $curtain): RedirectResponse
    {
        $data = $request->validate([
            'title'                       => 'required|max:191',
            'title_icon'                  => 'sometimes',
            'sliders'                     => 'array',
            'sliders.*.description'       => 'nullable|string',
            'sliders.*.bg_image'          => 'nullable|file|mimes:jpeg,png,jpg,gif,svg',
            'sliders.*.bg_video'          => 'nullable|file|mimes:mp4,webm,ogg',
            'sliders.*.bg_color'          => 'nullable|string',
            'sliders.*.title_color'       => 'nullable|string',
            'sliders.*.description_color' => 'nullable|string',
        ]);

        // Update basic curtain info
        $curtain->update([
            'title'       => $data['title'],
            'title_icon'  => $data['title_icon'] ?? null,
        ]);

        $processedSliders = [];

        if (isset($data['sliders']) && is_array($data['sliders'])) {
            foreach ($data['sliders'] as $sliderIndex => $slider) {
                $processedSlider = [
                    'title'             => $data['title'], // Use the curtain title
                    'description'       => $slider['description'] ?? '',
                    'bg_color'          => $slider['bg_color'] ?? '',
                    'title_color'       => $slider['title_color'] ?? '',
                    'description_color' => $slider['description_color'] ?? '',
                ];

                // Handle existing images/videos - preserve them if no new file uploaded
                $existingSliders = $curtain->sliders ?? [];

                // Handle background image
                if (isset($slider['bg_image']) && $slider['bg_image'] instanceof UploadedFile) {
                    $processedSlider['bg_image'] = '/uploads/' . $slider['bg_image']->store('curtains', 'uploads');
                } else {
                    // Keep existing image if available
                    $processedSlider['bg_image'] = $existingSliders[$sliderIndex]['bg_image'] ?? '';
                }

                // Handle background video
                if (isset($slider['bg_video']) && $slider['bg_video'] instanceof UploadedFile) {
                    $processedSlider['bg_video'] = '/uploads/' . $slider['bg_video']->store('curtains', 'uploads');
                } else {
                    // Keep existing video if available
                    $processedSlider['bg_video'] = $existingSliders[$sliderIndex]['bg_video'] ?? '';
                }

                $processedSliders[] = $processedSlider;
            }
        }

        // Update curtain with processed sliders
        $curtain->update(['sliders' => $processedSliders]);

        return back()->with([
            'type'    => 'success',
            'message' => __('The curtain was successfully updated'),
        ]);
    }
}
