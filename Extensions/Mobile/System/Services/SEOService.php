<?php

namespace App\Extensions\SEOTool\System\Services;

use App\Domains\Entity\Enums\EntityEnum;
use App\Domains\Entity\Facades\Entity;
use App\Extensions\SEOTool\System\Services\Search\SerperDevSearch;
use App\Helpers\Classes\ApiHelper;
use App\Helpers\Classes\Helper;
use App\Models\OpenAIGenerator;
use App\Models\UserOpenai;
use Exception;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Config;
use Illuminate\Support\Str;
use JsonException;
use OpenAI\Laravel\Facades\OpenAI;

class SEOService
{
    /**
     * @throws JsonException
     * @throws Exception
     */
    public static function getKeywords($topic): array
    {
        $driver = Entity::driver(EntityEnum::SERPER);
        if (! $driver->hasCreditBalance()) {
            return [__('You have no credits left. Please consider upgrading your plan.')];
        }

        $serperDev = new SerperDevSearch;
        $keywords = $serperDev->getKeywords($topic);
        $keywordsJson = json_encode($keywords, JSON_THROW_ON_ERROR);

        $driver->input($keywordsJson)->calculateCredit()->decreaseCredit();

        return $keywords;
    }

    /**
     * @throws Exception
     */
    public static function analiyzeWithAI(Request $request): array
    {
        $keyword = $request->topicKeyword;
        $serperDev = new SerperDevSearch;

        ApiHelper::setOpenAiKey();
        $default_model = Helper::setting('openai_default_model');
        $driver = Entity::driver(EntityEnum::fromSlug($default_model));
        if (! $driver->hasCreditBalance()) {
            return [__('You have no credits left. Please consider upgrading your plan.')];
        }
        $content = $request->resultText ?? '';
        $contentType = $request->type ?? 'article';

        $serperDriver = Entity::driver(EntityEnum::SERPER);
        if (! $serperDriver->hasCreditBalance()) {
            return [__('You have no credits left. Please consider upgrading your plan.')];
        }
        $googleResult = json_encode($serperDev->search($keyword), JSON_THROW_ON_ERROR);
        $serperDriver->input($googleResult)->calculateCredit()->decreaseCredit();

        $resultForCompatitorAndLongTailKeywords = OpenAI::chat()->create([
            'model'    => $default_model,
            'messages' => [
                [
                    'role'    => 'user',
                    'content' => self::getCompetitorAndLongTailKeywordsPrompt($googleResult, $request->type, $keyword),
                ],
            ],
        ]);
        $firsResult = $resultForCompatitorAndLongTailKeywords->choices[0]->message->content;
        $competitorKeywords = json_encode(self::extractData($firsResult)['competitorList'], JSON_THROW_ON_ERROR);
        $longTailKeywords = json_encode(self::extractData($firsResult)['longTailList'], JSON_THROW_ON_ERROR);
        $imagesCount = $request->imagesCount;
        $headersCount = $request->headersCount;
        $linksCount = $request->linksCount;

        $history = self::writeContentHistory($contentType, $imagesCount, $headersCount, $linksCount, $competitorKeywords, $longTailKeywords);
        $history[] = [
            'role'    => 'user',
            'content' => self::analyzeContentPrompt($content, $imagesCount, $headersCount, $linksCount, $googleResult, $competitorKeywords, $longTailKeywords, $contentType),
        ];
        $result = OpenAI::chat()->create([
            'model'    => $default_model,
            'messages' => $history,
        ]);
        $secondResult = $result->choices[0]->message->content;
        $driver->input($firsResult . $secondResult)->calculateCredit()->decreaseCredit();

        return self::extractData($secondResult, $competitorKeywords, $longTailKeywords);
    }

    /**
     * @throws JsonException
     * @throws Exception
     */
    public static function improveWithAI(Request $request)
    {
        $keyword = $request->topicKeyword;
        $serperDev = new SerperDevSearch;
        ApiHelper::setOpenAiKey();
        $default_model = Helper::setting('openai_default_model');
        $driver = Entity::driver(EntityEnum::fromSlug($default_model));
        $driver->redirectIfNoCreditBalance();
        $content = $request->resultText ?? '';
        $contentType = $request->type ?? 'article';
        $googleResult = json_encode($serperDev->search($keyword), JSON_THROW_ON_ERROR);
        $competitorList = $request->competitorList;
        $longTailList = $request->longTailList;
        $imagesCount = $request->imagesCount;
        $headersCount = $request->headersCount;
        $linksCount = $request->linksCount;
        session_start();
        header('Content-type: text/event-stream');
        header('Cache-Control: no-cache');
        $result = OpenAI::chat()->createStreamed([
            'model'    => $default_model,
            'messages' => [
                [
                    'role'    => 'user',
                    'content' => self::improveContentPrompt($content, $imagesCount, $headersCount, $linksCount, $googleResult, $longTailList, $competitorList, $contentType),
                ],
            ],
            'stream' => true,
        ]);

        $driver->input($result)->calculateCredit()->decreaseCredit();
        $total_used_tokens = 0;
        $output = '';
        $responsedText = '';
        foreach ($result as $response) {
            $message = $response->choices[0]->delta->content;
            $messageFix = str_replace(["\r\n", "\r", "\n"], '<br/>', $message);
            $output .= $messageFix;
            $responsedText .= $message;
            $total_used_tokens += countWords($messageFix);

            echo "event: data\n";
            echo 'data: ' . $messageFix . "\n\n";
            flush();
        }
        echo "event: stop\n";
        echo "data: [DONE]\n\n";

        if ($request->seoTool) {
            $user = Auth::user();
            $post = OpenAIGenerator::where('slug', 'ai_article_wizard_generator')->first();
            $entry = new UserOpenai;
            $entry->team_id = $user->team_id;
            $entry->title = $keyword ?: __('New Workbook');
            $entry->slug = str()->random(7) . str($user->fullName())->slug() . '-workbook';
            $entry->user_id = $user->id;
            $entry->openai_id = $post->id;
            $entry->input = $content;
            $entry->response = $responsedText;
            $entry->output = $output;
            $entry->hash = Str::random(256);
            $entry->credits = $total_used_tokens;
            $entry->words = 0;
            $entry->save();
        }
    }

    /**
     * @throws JsonException
     */
    public static function generateSEO(Request $request)
    {
        $prompt = 'Generate SEO for the following:';
        switch ($request->input) {
            case 'meta_title':
                $prompt .= ' Meta Title';
                $keyword = 'AI Tools';

                break;
            case 'meta_description':
                $prompt .= ' Meta Description';
                $keyword = 'AI Tools';

                break;
            case 'meta_keywords':
                $prompt .= ' Meta Keywords';
                $keyword = 'AI Tools';

                break;
            case 'seo_description':
                $prompt .= ' SEO Blog Post Description';
                $keyword = $request->keyword ?? 'AI Tools';

                break;
            case 'seo_title':
                $prompt .= ' SEO Blog Post Title';
                $keyword = $request->keyword ?? 'AI Tools';

                break;

            default:
                $prompt .= ' Meta Title, Meta Description, Meta Keywords';
                $keyword = 'AI Tools';

                break;
        }
        $serperDev = new SerperDevSearch;
        $driver = Entity::driver(EntityEnum::SERPER);
        $driver->redirectIfNoCreditBalance();
        $google_resault = json_encode($serperDev->search($keyword), JSON_THROW_ON_ERROR);
        $driver->input($google_resault)->calculateCredit()->decreaseCredit();
        $app_name = Config::get('app.name');
        $prompt .= " for $app_name using the following google search results: ";
        $prompt .= $google_resault;
        $prompt .= '. The output should be text with comma separated values.';

        $default_model = Helper::setting('openai_default_model');
        $driverOpenai = Entity::driver(EntityEnum::fromSlug($default_model));
        $driverOpenai->redirectIfNoCreditBalance();

        ApiHelper::setOpenAIKey();
        $completion = OpenAI::chat()->create([
            'model'    => $default_model,
            'messages' => [[
                'role'    => 'user',
                'content' => $prompt,
            ]],
        ]);
        $result = $completion['choices'][0]['message']['content'];
        $driverOpenai->input($result)->calculateCredit()->decreaseCredit();

        return $result;
    }

    /**
     * @throws Exception
     */
    public static function getSearchQuestions(Request $request)
    {
        $serperDev = new SerperDevSearch;
        $driver = Entity::driver(EntityEnum::SERPER);
        $driver->redirectIfNoCreditBalance();
        $questions = $serperDev->getPeopleAlsoAsks($request->title);
        $stringQuestions = collect($questions)->map(function ($question) {
            return json_encode($question, JSON_THROW_ON_ERROR);
        });
        $driver->input($stringQuestions)->calculateCredit()->decreaseCredit();

        return $stringQuestions;
    }

    // Helper function to extract data from the AI response

    /**
     * @throws JsonException
     */
    private static function extractData($text, $competitorKeywords = '', $longTailKeywords = '')
    {
        // Initialize an array to hold the extracted data
        $data = [
            'percentage'     => 70,
            'competitorList' => [],
            'longTailList'   => [],
        ];

        // Define regex patterns to capture the data
        $patterns = [
            'percentage'     => '/\s*\[(\d+(\.\d+)?)%\]/',
            'competitorList' => '/competitorList:\s*\[([^\]]+)\]/',
            'longTailList'   => '/longTailList:\s*\[([^\]]+)\]/',
        ];

        // Extract percentage
        if (preg_match($patterns['percentage'], $text, $matches)) {
            $data['percentage'] = $matches[1];
        }

        if (! empty($competitorKeywords)) {
            $data['competitorList'] = json_decode($competitorKeywords, false, 512, JSON_THROW_ON_ERROR);
        } elseif (preg_match($patterns['competitorList'], $text, $matches)) {
            $data['competitorList'] = array_map('trim', explode(',', $matches[1]));
        }

        if (! empty($longTailKeywords)) {
            $data['longTailList'] = json_decode($longTailKeywords, false, 512, JSON_THROW_ON_ERROR);
        } elseif (preg_match($patterns['longTailList'], $text, $matches)) {
            $data['longTailList'] = array_map('trim', explode(',', $matches[1]));
        }

        // Return the extracted data
        return $data;
    }

    private static function improveContentPrompt($content, $imagesCount, $headersCount, $linksCount, $googleSearchResult, $longTailKeywords, $competitorKeywords, $type = 'article')
    {
        $article = '
		[' . $content . ']
		';

        $prompt = '
		Please review the following ' . $type . ' and improve it based on the specified SEO factors. Ensure that the ' . $type . ' meets all the SEO criteria listed below. Provide a detailed explanation of the changes made and why they were necessary to enhance the ' . $type . "'s SEO performance.

		SEO Factors:

		1. Title Tag (H1) Presence at the Top of the " . $type . '
		- Ensure the main title is an H1 tag and placed at the top of the ' . $type . '.
		2. Title Length
		- The title should be between 50-60 characters for optimal length.
		3. Meta Description
		- Include a meta description of 150-160 characters summarizing the ' . $type . ' content.
		4. Headings (H2, H3, etc.)
		- Use subheadings (H2, H3, etc.) to organize content and make it easier to read. Current headings count: ' . $headersCount . '
		5. Primary Keyword Density
		- Ensure primary keywords are used naturally throughout the ' . $type . ' (ideal density: 1-2%).
		6. LSI Keywords
		- Include Latent Semantic Indexing (LSI) keywords to provide context and relevance.
		7. Word Count
		- The ' . $type . ' should be at least 1000 words long to cover the topic comprehensively.
		8. Readability
		- Ensure the ' . $type . ' is easy to read, aiming for a Flesch-Kincaid readability score appropriate for the target audience.
		9. External Images with Alt Tags
		- Include relevant images with descriptive alt tags. Current images count: ' . $imagesCount . '
		10. Links
			- Include at least 3-5 internal links and 2-3 external links to authoritative sources. Current links count: ' . $linksCount . '
		11. SEO-Friendly URL
			- Ensure the URL is short, descriptive, and includes the primary keyword.
		12. Internal Links
			- Add links to other relevant pages within the same website.
		13. Schema Markup
			- Ensure the use of appropriate schema markup to enhance search engine understanding.
		14. User Engagement Metrics
			- Aim to improve metrics such as time on page and reduce bounce rate by making content engaging.
		15. Canonical Tags
			- Use canonical tags to prevent duplicate content issues.
		16. Competitor Keywords Should be included in the content.
			- Identify and list competitor keywords related to the ' . $type . '.
		17. Long-Tail Keywords Should be included in the content.
			- Identify and list long-tail keywords related to the ' . $type . '.

		' . $type . " to Review and Improve:

		$article

		Google recent search result for the " . $type . ' Topic:
    	' . $googleSearchResult . '.

		Long-Tail Keywords:
		[' . $longTailKeywords . '].

		Competitor Keywords:
		[' . $competitorKeywords . '].

		Instructions:

		1. Review the ' . $type . ' thoroughly against each SEO factor listed above.
		2. Make sure to add all possible competitor and long-tail keywords provided above to the content to improve it.
		3. Make necessary changes to improve the ' . $type . "'s SEO performance.
		4. Return only HTML (without head and body) for the new SEO improved " . $type . ' content.
		5. Must not write ```html in the response.
		6. If you going to add any images, please provide the image full valid URL starting with https:// not relative URL.
		';

        return $prompt;
    }

    private static function analyzeContentPrompt($content, $imagesCount, $headersCount, $linksCount, $googleSearchResult, $longTailKeywords, $competitorKeywords, $type = 'article')
    {
        $prompt = '
		Please analyze the following ' . $type . ' for its compatibility with SEO factors. Evaluate how well the ' . $type . ' adheres to the specified SEO criteria listed below.

		SEO Factors:

		1. Title Tag (H1) Presence at the Top of the ' . $type . '
		- Ensure the main title is an H1 tag and placed at the top of the ' . $type . '.
		2. Title Length
		- The title should be between 50-60 characters for optimal length.
		3. Meta Description
		- Include a meta description of 150-160 characters summarizing the ' . $type . ' content.
		4. Headings (H2, H3, etc.)
		- Use subheadings (H2, H3, etc.) to organize content and make it easier to read. Current headings count: ' . $headersCount . '
		5. Primary Keyword Density
		- Ensure primary keywords are used naturally throughout the ' . $type . ' (ideal density: 1-2%).
		6. LSI Keywords
		- Include Latent Semantic Indexing (LSI) keywords to provide context and relevance.
		7. Word Count
		- The ' . $type . ' should be at least 1000 words long to cover the topic comprehensively.
		8. Readability
		- Ensure the ' . $type . ' is easy to read, aiming for a Flesch-Kincaid readability score appropriate for the target audience.
		9. External Images with Alt Tags
		- Include relevant images with descriptive alt tags. Current images count: ' . $imagesCount . '
		10. Links
			- Include at least 3-5 internal links and 2-3 external links to authoritative sources. Current links count: ' . $linksCount . '
		11. SEO-Friendly URL
			- Ensure the URL is short, descriptive, and includes the primary keyword.
		12. Internal Links
			- Add links to other relevant pages within the same website.
		13. Schema Markup
			- Ensure the use of appropriate schema markup to enhance search engine understanding.
		14. User Engagement Metrics
			- Aim to improve metrics such as time on page and reduce bounce rate by making content engaging.
		15. Canonical Tags
			- Use canonical tags to prevent duplicate content issues.
		16. Competitor Keywords
			- Identify and list competitor keywords related to the ' . $type . '.
		17. Long-Tail Keywords
			- Identify and list long-tail keywords related to the ' . $type . '.

		' . $type . ' to Analyze:

		[' . $content . ']

		Google recent search result for the ' . $type . ' Topic:
    	' . $googleSearchResult . '.

		Long-Tail Keywords:
		[' . $longTailKeywords . '].

		Competitor Keywords:
		[' . $competitorKeywords . '].

		Instructions:

		1. Analyze the ' . $type . ' against each SEO factor listed above, specifically focusing on Competitor and Long-Tail Keywords. Dont be productive, if the factor present in the content, just add points to score.
		2. Based on Google recent search results, support your analysis with relevant data.
		3. Provide a percentage score indicating the compatibility of the ' . $type . ' with SEO criteria. Make sure the percentage is accurate and reflects the ' . $type . "'s SEO performance based on the factors evaluated correctly.
		4. Return only and only percentage score as the analysis result and the percentage format should be '[percentage%]'.";

        return $prompt;
    }

    private static function writeContentHistory($type, $imagesCount, $headersCount, $linksCount, $longTailKeywords, $competitorKeywords)
    {
        $exampleContent = '
			<h1>Tesla Cars: Model 3, Model Y, and Model X - A Comprehensive Comparison</h1>
			<div><br><br></div>
			<h2>Tesla Model 3 vs. Model Y</h2>
			<div>&nbsp;</div>
			<p>When considering Teslas lineup, the <a href="https://www.tesla.com/model3" target="_blank" rel="noopener">Model 3</a> and <a href="https://www.tesla.com/modely" target="_blank" rel="noopener">Model Y</a> often draw comparisons. The Model 3, a compact sedan, offers a sportier drive and is generally more affordable, making it Teslas most popular vehicle. On the other hand, the Model Y, an SUV, provides more cargo space, higher seating, and additional versatility. Both models come with impressive ranges and advanced <a href="https://www.tesla.com/autopilot" target="_blank" rel="noopener">autopilot features</a>, but the choice ultimately depends on whether buyers prioritize efficiency or practicality.</p>
			<div><br><br></div>
			<h2>Top Reasons to Buy a Tesla</h2>
			<div>&nbsp;</div>
			<p>Tesla vehicles offer numerous advantages that make them attractive to consumers. Key reasons include cutting-edge technology, zero-emission electric powertrains, and a robust <a href="https://www.tesla.com/supercharger" target="_blank" rel="noopener">Supercharger network</a> for long-distance travel. Additionally, Teslas over-the-air software updates ensure the cars systems and features continually improve, keeping the vehicle up-to-date long after purchase. Lastly, the brands strong focus on innovation and sustainability appeals to environmentally conscious buyers.</p>
			<div><br><br></div>
			<h2>Tesla Model X: Price and Features</h2>
			<div>&nbsp;</div>
			<p>The <a href="https://www.tesla.com/modelx" target="_blank" rel="noopener">Tesla Model X</a> stands out with its distinctive falcon-wing doors and spacious interior. Its price starts at $89,990, reflecting its premium nature. It includes features such as a potent dual-motor all-wheel drive, advanced autopilot, and one of the fastest accelerations for an SUV. The Model X can seat up to seven passengers and provides ample cargo space, making it ideal for families. Its long range, coupled with Teslas Supercharger network, ensures that long trips are feasible and convenient.</p> ';
        $exampleGoogleResult = '{
			"searchParameters": {
				"q": "Tesla Cars: Model 3, Model Y, and Model X - A Comprehensive Comparison",
				"type": "search",
				"engine": "google"
			},
			"organic": [
				{
				"title": "Compare | Tesla",
				"link": "https://www.tesla.com/compare",
				"snippet": "Compare the pricing and specifications of Model S, Model 3, Model X and Model Y to find the right Tesla for you.",
				"attributes": {
					"Missing": "Comprehensive | Show results with:Comprehensive"
				},
				"position": 1
				},
				{
				"title": "Tesla Model Y vs. Model 3 vs. Model X vs. Model S (Long Range ...",
				"link": "https://cleantechnica.com/2020/04/18/tesla-model-y-vs-model-3-vs-model-x-vs-model-s-long-range-performance-trims/",
				"snippet": "Each model (by far) holds the best-in-class electric range. In the EV space, it can be argued that Tesla has really been in a league of its own.",
				"date": "Apr 18, 2020",
				"attributes": {
					"Missing": "Comprehensive | Show results with:Comprehensive"
				},
				"position": 2
				},
				{
				"title": "Tesla Model 3 vs Tesla Model Y - Which Should You Get? - YouTube",
				"link": "https://www.youtube.com/watch?v=YX_xwsbxGZU",
				"snippet": "As an owner of both, Model Y is the way to go. The extra space makes such a quality of life ...",
				"date": "Nov 13, 2023",
				"attributes": {
					"Duration": "27:28",
					"Posted": "Nov 13, 2023"
				},
				"imageUrl": "https://i.ytimg.com/vi/YX_xwsbxGZU/default.jpg?sqp=-oaymwEECHgQQw&rs=AMzJL3nmS2TVm_A-UUY-WsF9rLFotOw9Bw",
				"position": 3
				},
				{
				"title": "What is better in a model Y vs a model X if anything? : r/TeslaModelX",
				"link": "https://www.reddit.com/r/TeslaModelX/comments/17098f6/what_is_better_in_a_model_y_vs_a_model_x_if/",
				"snippet": "The X is a comfy large SUV/MPV, the Y is a sporty mid size SUV. Theyre aimed at different things. The Y is the perfect SUV for the european ...",
				"date": "Oct 5, 2023",
				"attributes": {
					"Missing": "Comprehensive | Show results with:Comprehensive"
				},
				"sitelinks": [
					{
					"title": "Model Y vs Model X : r/TeslaLounge - Reddit",
					"link": "https://www.reddit.com/r/TeslaLounge/comments/14c2stf/model_y_vs_model_x/"
					},
					{
					"title": "Model Y vs Model 3 : r/TeslaModelY - Reddit",
					"link": "https://www.reddit.com/r/TeslaModelY/comments/18g08x5/model_y_vs_model_3/"
					},
					{
					"title": "Considering Model X or Model Y : r/TeslaLounge - Reddit",
					"link": "https://www.reddit.com/r/TeslaLounge/comments/178nhyh/considering_model_x_or_model_y/"
					},
					{
					"title": "Struggling trying to choose between Model Y and Model X - Reddit",
					"link": "https://www.reddit.com/r/TeslaLounge/comments/16n05nw/struggling_trying_to_choose_between_model_y_and/"
					}
				],
				"position": 4
				},
				{
				"title": "Tesla cheat sheet: Model 3 vs Model Y vs Model X vs Model S",
				"link": "https://www.tomsguide.com/news/tesla-model-3-vs-model-y-vs-model-x-vs-model-s",
				"snippet": "Our Tesla Model 3 vs Model Y vs Model X vs Model S comparison shows that there are plenty of options and tiers. And these packages arent just ...",
				"date": "Mar 15, 2022",
				"attributes": {
					"Missing": "Comprehensive | Show results with:Comprehensive"
				},
				"position": 5
				},
				{
				"title": "Tesla Model 3 vs. Tesla Model Y: which is better? - Cinch",
				"link": "https://www.cinch.co.uk/guides/choosing-a-car/tesla-model-3-vs-tesla-model-y",
				"snippet": "Read our guide for a helpful comparison of the Tesla Model 3 hatchback and Tesla Model Y small SUV. Decide which is best for you with our help.",
				"position": 6
				},
				{
				"title": "Tesla Model X vs. Tesla Model Y: Whats the difference? - Toms Guide",
				"link": "https://www.tomsguide.com/news/tesla-model-x-vs-tesla-model-y",
				"snippet": "The Tesla Model Y first appeared in 2020, so while its newer the other bonus is its leaner looks and slightly more refined feel compared to ...",
				"date": "May 16, 2024",
				"attributes": {
					"Missing": "Comprehensive | Show results with:Comprehensive"
				},
				"position": 7
				},
				{
				"title": "How to tell the difference between a Tesla Model X and ... - Quora",
				"link": "https://www.quora.com/How-can-I-tell-the-difference-between-a-Tesla-Model-X-and-Model-Y-They-both-have-the-4-door-fastback-look-but-sport-no-badges-with-the-model-name-Does-anyone-know-an-easy-way-to-tell-them-apart",
				"snippet": "The Model X is a larger car and based off of the Model S. The Model Y is based off of the Model 3. From the front the model X will resemble a ...",
				"date": "Apr 11, 2022",
				"attributes": {
					"Missing": "Comprehensive | Show results with:Comprehensive"
				},
				"position": 8
				},
				{
				"title": "Tesla Model Y And Model 3 Comparison and Why They Are So ...",
				"link": "https://www.youtube.com/watch?v=v3YvzrmH2mY",
				"snippet": "Check out this video for a summary of similarities and differences between the Model Y and ...",
				"date": "Jul 14, 2023",
				"attributes": {
					"Duration": "19:18",
					"Posted": "Jul 14, 2023"
				},
				"imageUrl": "https://i.ytimg.com/vi/v3YvzrmH2mY/default.jpg?sqp=-oaymwEECHgQQw&rs=AMzJL3lmaJ5pEwOrgW4TXfWdIBVCh95Ngg",
				"position": 9
				},
				{
				"title": "2024 Tesla Model 3 vs. 2023 Tesla Model Y: Y U Should Wait",
				"link": "https://www.motortrend.com/reviews/2024-tesla-model-3-vs-2023-tesla-model-y-comparison-test-review/",
				"snippet": "Purely from a driving perspective, there are fewer differences. The Model 3s new suspension geometry isnt obvious through the steering wheel ...",
				"date": "Nov 10, 2023",
				"position": 10
				}
			],
			"peopleAlsoAsk": [
				{
				"question": "Which Tesla is better, X or Y?",
				"snippet": "The Model Y can travel between 303 and 330 miles on a charge depending on your choice of the Performance or Long Range trims, according to Tesla. The Model X, despite being larger and heavier, fits a larger battery and can travel between 311 and 348 miles per charge depending on which model you choose.",
				"title": "Tesla Model X vs. Tesla Model Y Specs Comparison - MotorTrend",
				"link": "https://www.motortrend.com/features/tesla-model-x-specs-tesla-model-y-specs/"
				},
				{
				"question": "Which Tesla is better, 3 or Y?",
				"snippet": "The Model 3 Long Range also has slighly better range than the Model Y Long Range. The standard Model Y has slightly better than the standard Model 3, though. Price: The Model Y is generally more expensive than the Model 3, reflecting its larger size and additional features like available third-row seating.",
				"title": "Tesla Model 3 vs Model Y: Whats the Difference? | AutoNation",
				"link": "https://www.autonation.com/vehicle-research/tesla-model-3-vs-model-y"
				},
				{
				"question": "Is a Tesla Model X better than a Model 3?",
				"snippet": "Tesla Model 3. When comparing the Tesla Model Xs and the Tesla Model 3s specifications and ratings, the Tesla Model X has the advantage in the areas of reliability, overall quality score and base engine power. The Tesla Model 3 has the advantage in the area of resale value.",
				"title": "Tesla Model X vs. Tesla Model 3 - iSeeCars.com",
				"link": "https://www.iseecars.com/compare/tesla-model_x-vs-tesla-model_3"
				},
				{
				"question": "Is it worth upgrading from Model 3 to Model Y?",
				"snippet": "Thanks to its lower and lighter body, the Model 3 is slightly more fun to drive and faster when compared to an equivalent Model Y variant. Bear in mind that the differences are not significant, so if you want maximum space, comfort, and 95% of the Model 3 driving experience, the Model Y is clearly the one to get.",
				"title": "Tesla Model 3 Vs. Model Y: Does Upgrading To The Crossover Make ...",
				"link": "https://insideevs.com/news/697348/tesla-model-3-vs-model-y-does-upgrading-to-crossover-make-sense/"
				}
			],
			"relatedSearches": [
				{
				"query": "Tesla Model 3"
				},
				{
				"query": "Tesla Model Y"
				},
				{
				"query": "Tesla Model X"
				},
				{
				"query": "Tesla Model Y price"
				},
				{
				"query": "Tesla models and prices"
				},
				{
				"query": "2024 Tesla Model Y MSRP"
				},
				{
				"query": "Tesla Model 3 price"
				},
				{
				"query": "Tesla Model 3 price in USA"
				}
			]
		}';
        $history[] = [
            'role'    => 'system',
            'content' => 'You are a helpful SEO analysis. You receive a ' . $type . ' and a google search results for that ' . $type . ' topic.',
        ];
        $history[] = [
            'role'    => 'user',
            'content' => self::analyzeContentPrompt($exampleContent, $imagesCount, $headersCount, $linksCount, $longTailKeywords, $competitorKeywords, $exampleGoogleResult, $type),
        ];
        $history[] = [
            'role'    => 'assistant',
            'content' => '
				Analyzing the provided content against each SEO factor:
				1. **Title Tag (H1) Presence at the Top of the [object Object]**
				- Yes, the main title is an H1 tag and placed at the top.
				2. **Title Length**
				- The title "Tesla Cars: Model 3, Model Y, and Model X - A Comprehensive Comparison" is 57 characters long. This falls within the optimal range of 50-60 characters.
				3. **Meta Description**
				- No meta description is provided in the provided content.
				4. **Headings (H2, H3, etc.)**
				- Subheadings (H2) are used to organize the content: "Tesla Model 3 vs. Model Y", "Top Reasons to Buy a Tesla," and "Tesla Model X: Price and Features."
				5. **Primary Keyword Density**
				- Primary keywords like "Tesla", "Model 3", "Model Y", and "Model X" are used throughout the content, but actual density cannot be determined with this text alone.
				6. **LSI Keywords**
				- LSI keywords such as "electric powertrains", "autopilot features", and "Supercharger network" are included, adding context and relevance.
				7. **Word Count**
				- The provided content contains 548 words, which is below the recommended 1000-word count for comprehensive coverage.
				8. **Readability**
				- The content appears to be easily readable, but an exact Flesch-Kincaid score cannot be determined from this text alone.
				9. **External Images with Alt Tags**
				- No images are included in the provided content.
				10. **Links**
					- There are at least 3 external links to authoritative sources, but no internal links are provided.
				11. **SEO-Friendly URL**
					- The analysis does not include the URL, so this factor cannot be assessed.
				12. **Internal Links**
					- No internal links are provided in the content.
				13. **Schema Markup**
					- The analysis does not include schema markup data, so this factor cannot be assessed.
				14. **User Engagement Metrics**
					- User engagement metrics cannot be determined from the content alone.
				15. **Canonical Tags**
					- The analysis does not include canonical tags, so this factor cannot be assessed.
				16. **Competitor Keywords**
					- Competitor keywords related to the "Tesla Cars: Model 3, Model Y, and Model X - A Comprehensive Comparison" topic include "Tesla Model 3 vs. Model Y vs. Model X vs. Model S", "Tesla Model Y vs. Model 3 vs. Model X vs. Model S", and "Tesla Model 3 vs Tesla Model Y - Which Should You Get?".
				17. **Long-Tail Keywords**
					- Long-tail keywords related to the "Tesla Cars: Model 3, Model Y, and Model X - A Comprehensive Comparison" topic include "Tesla Model 3 vs. Model Y", "Top Reasons to Buy a Tesla", and "Tesla Model X: Price and Features".
				### Evaluation Summary:
				- Presence of title at the top: Yes (100%)
				- Title length: Yes (100%)
				- Meta description: No (0%)
				- Headings used: Yes (100%)
				- Primary keyword density: Potentially (75%)
				- LSI keywords: Yes (100%)
				- Word count: No (50%)
				- Readability: Yes (100%)
				- External images with alt tags: No (0%)
				- Links (internal and external): Partial (50%)
				- SEO-friendly URL: Not rated (N/A)
				- Internal links: No (0%)
				- Schema markup: Not rated (N/A)
				- User engagement: Not rated (N/A)
				- Canonical tags: Not rated (N/A)
				- Competitor keywords: Identified matched keywords (40%)
				- Long-tail keywords: Identified matched keywords (10%)
				### Compatibility Score Calculation:
				1. Title at the top: 100%
				2. Title length: 100%
				3. Meta description: 0%
				4. Headings: 100%
				5. Primary keyword density: 75%
				6. LSI keywords: 100%
				7. Word count: 50%
				8. Readability: 100%
				9. External images with alt tags: 0%
				10. Links (internal and external): 50%
				11. SEO-friendly URL: N/A (0%)
				12. Internal links: 0%
				13. Schema markup: N/A (0%)
				16. User engagement: N/A (0%)
				17. Canonical tags: N/A (0%)
				18. Competitor keywords: 40%
				19. Long-tail keywords: 10%
				Total points achieved (without N/A factors considered): 725% / 1100% = 65.9%
				### Final Compatibility Score:
				[65.9%]',
        ];
        $history[] = [
            'role'    => 'user',
            'content' => 'Do the same logic for analyzing the content but return only Final Compatibility Score values, .',
        ];
        $history[] = [
            'role'    => 'assistant',
            'content' => '
				percentage: [65.9%]
				competitorList: [model y vs model 3, model 3 vs model y, tesla model y vs model 3, tesla model comparison, tesla models, tesla prices, model s vs model 3]
				longTailList: [tesla model y comparison, tesla model y size comparison to other suvs, model y color comparison, model y performance 0-60 comparison, tesla model y comparison review]
			',
        ];

        return $history;
    }

    private static function getCompetitorAndLongTailKeywordsPrompt($googleSearchResult, $type, $keyword)
    {
        $prompt = '
		Google recent search result for the ' . $type . ' Topic:
    	' . $googleSearchResult . '

		Instructions:
		1. Analyze the Google search results for the ' . $type . ' topic.
		2. Identify competitor keywords related to the main keyword: ' . $keyword . '.
		3. Return a list of competitor keywords that are commonly associated with the ' . $type . " topic.
		4. The list should include at least 5 competitor keywords.
		5. Return the competitor keywords in the following format:
		'competitorList: [competitorKeyword 1, competitorKeyword 2, competitorKeyword 3, competitorKeyword 4, competitorKeyword 5, ..]'
		6. Identify long-tail keywords related to the main keyword: " . $keyword . '.
		7. Return a list of long-tail keywords that are commonly associated with the ' . $type . " topic.
		8. The list should include at least 5 long-tail keywords.
		9. Return the long-tail keywords in the following format:
		'longTailList: [longTailKeyword 1, longTailKeyword 2, longTailKeyword 3, longTailKeyword 4, longTailKeyword 5, ..]'
		10. The response format should only include both competitor and long-tail keywords as the following example:
		competitorList: [competitorKeyword 1, competitorKeyword 2, competitorKeyword 3, competitorKeyword 4, competitorKeyword 5, ..]
		longTailList: [longTailKeyword 1, longTailKeyword 2, longTailKeyword 3, longTailKeyword 4, longTailKeyword 5, ..]'";

        return $prompt;
    }
}
