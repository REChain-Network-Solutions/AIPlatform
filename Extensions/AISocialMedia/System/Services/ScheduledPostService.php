<?php

namespace App\Extensions\AISocialMedia\System\Services;

use App\Extensions\AISocialMedia\System\Models\ScheduledPost;
use App\Models\Company;
use App\Models\Product;
use Illuminate\Database\Eloquent\Builder;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Fluent;

class ScheduledPostService
{
    public function __construct(
        public AutomationService $automationService
    ) {}

    public function store(array $data)
    {
        $data = $this->data($data);

        return ScheduledPost::query()->create($data);
    }

    public function data(array $data): array
    {
        $automationPlatform = $this->automationService->findAutomationPlatform($data['platform_id']);

        $data = new Fluent($data);

        $campaignTarget = str_replace("\r\n", ', ', $data['camp_target']);

        $productString = $this->productString($data['product_id']);

        $data['topics'] = is_array($data['topics']) ? implode(', ', $data['topics']) : $data['topics'];

        return [
            'user_id'                       => Auth::id(),
            'review'                        => (bool) ($data['review']),
            'is_email'                      => (bool) ($data['sendMail']),
            'auto_generate'                 => (bool) ($data['auto_generate']),
            'is_seo'                        => (bool) ($data['seo']),
            'is_img'                        => (bool) ($data['is_img']),
            'is_repeated'                   => (bool) ($data['repeat']),
            'tone'                          => $data['tone'],
            'num_res'                       => $data['num_res'],
            'vis_format'                    => $data['vis_format'],
            'vis_ratio'                     => $data['vis_ratio'],
            'time'                          => $data['time'],
            'repeat_period'                 => $data['repeat_period'],
            'repeat_start_date'             => date('Y-m-d', strtotime($data['date'])),
            'repeat_time'                   => $data['time'],
            'company_id'                    => $data['company_id'],
            'platform'                      => $data['platform_id'],
            'content'                       => $data['content'],
            'campaign_name'                 => $data['cam_injected_name'],
            'campaign_target'               => $campaignTarget,
            'length'                        => $data['num_res'],
            'topics'                        => $data['topics'],
            'prompt'                        => $this->generatePrompt($data, $productString),
            'products'                      => $productString,
            'visual_format'                 => $data['vis_format'],
            'visual_ratio'                  => $data['vis_ratio'],
            'media'                         => $data['image'],
            'automation_platform_id'        => $data['platform']['setting']?->id,
        ];
    }

    public function generatePrompt($data, $productString): string
    {
        $company = $this->company($data['company_id']);

        $companyName = $company?->getAttribute('name');
        $companyWebsite = $company?->getAttribute('website');

        $seo = $data['seo'] ? 'SEO optimized. ' : '';

        $companyNameString = $companyName ? 'by ' . $companyName . '. ' : '';

        $campaignTarget = str_replace("\r\n", ', ', $data['camp_target']);

        $campaignTargetString = $campaignTarget ? 'The campaign aims to reach: [' . $campaignTarget . ']. ' : '';

        $camInjectedNameString = $data['cam_injected_name'] ? 'Campaign Name: ' . $data['cam_injected_name'] . '. ' : '';

        $topicsString = $data['topics'] ? 'Topics: ' . $data['topics'] . '. ' : '';

        $toneString = $data['tone'] ? 'Tone of Voice: ' . $data['tone'] . '. ' : '';

        $companyWebsiteString = $companyWebsite ? 'other than the website.' : '.';

        return <<<TEXT
				Craft a with a maximum length of {$data['num_res']} characters, without any emojis, emoticons a text compelling social media post for {$data['platform']['name']} platform, to promote a
				$seo campaign $companyNameString
				Focus on this provided: $campaignTargetString [$productString]
				{$camInjectedNameString}
				{$topicsString}
				{$toneString}
				Do not include or links $companyWebsiteString.
				Must not ever increase the length of the post.
				TEXT;
    }

    public function productString($products): string
    {
        $products = Product::query()->whereIn('id', $products)->get();

        $productDescriptions = $products->map(function ($product, $index) {
            $typeDescription = match ($product->type) {
                0       => 'product' . ($index + 1), // Product
                1       => 'service' . ($index + 1), // Service
                default => ($index + 1),       // Other
            };

            return '(' . $typeDescription . ': ' . $product->name . ')';
        });

        return $productDescriptions->implode(', ');
    }

    public function company(int $id)
    {
        return Company::query()->firstWhere('id', $id);
    }

    public function query(): Builder
    {
        return ScheduledPost::query();
    }
}
