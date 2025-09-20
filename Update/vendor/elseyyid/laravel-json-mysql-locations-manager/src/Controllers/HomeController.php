<?php namespace Elseyyid\LaravelJsonLocationsManager\Controllers;

use App\Http\Controllers\Controller;
use Illuminate\Http\Request;
use \Elseyyid\LaravelJsonLocationsManager\Models\Strings;
use Illuminate\Filesystem\Filesystem;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Facades\Artisan;
use Elseyyid\LaravelJsonLocationsManager\Requests\SearchFormRequest;
use Elseyyid\LaravelJsonLocationsManager\Requests\NewStringFormRequest;
use Elseyyid\LaravelJsonLocationsManager\Requests\NewLangFormRequest;

class HomeController extends Controller
{
    public function __construct(Filesystem $filesystem)
    {
        $this->files = $filesystem;
    }

    public function index()
    {
        if (Schema::hasTable('strings')) {
			$fields = \DB::getSchemaBuilder()->getColumnListing('strings');
		} else {
			Artisan::call('elseyyid:location:install');
			$fields = \DB::getSchemaBuilder()->getColumnListing('strings');
		}
        $exceptions = ['en','code','created_at','updated_at'];
        $filtered = collect($fields)->filter(function ($value, $key) use($exceptions){
            if (!in_array($value,$exceptions) ) {
                return $value;
            }
        });
        return view('langs::home')->with('langs', $filtered);
    }

    public function lang($lang)
    {
        $list = Strings::select(['code','en',$lang])->get();
        return view('langs::list')->with('lang', $lang)->with('list', $list);
    }

    public function update(Request $request,$code)
    {
        $column_name = $request->get('name');
        $column_value = $request->get('value');
        if($request->has('name') && $request->has('value')) {
            $test = Strings::select()
                ->where('code', '=', $code)
                ->update([$column_name => $column_value]);
            return response()->json([ 'code'=>200], 200);
        }
        return response()->json([ 'error'=> 400, 'message'=> 'Not enought params' ], 400);
    }

    public function generateJson($lang)
    {
        $list = Strings::pluck($lang,'en');
        $json = json_encode_prettify($list);
        $this->files->put(base_path('lang/'.$lang.'.json'),$json);
        return redirect()->back()->with(config('elseyyid-location.message_success_variable'), 'Publicado');
    }

    public function newLang(NewLangFormRequest $request)
    {
        $fields = \DB::getSchemaBuilder()->getColumnListing('strings');
        if (! in_array( $request->newLang, $fields )) {
            Schema::table('strings', function (Blueprint $table) use($request){
                    $table->text($request->newLang)->nullable();
                });
        }
        return redirect()->route('elseyyid.translations.lang',$request->newLang)->with(config('elseyyid-location.message_success_variable'), 'Language '.$request->newLang. ' Created!');
    }

    public function newString(NewStringFormRequest $request)
    {
        $string = Strings::where('en',$request->newString)->first();
        if (!isset($string->code)) {
            Strings::create(['en' => $request->newString]);
        }
        return redirect()->back()->with(config('elseyyid-location.message_success_variable'), 'String '.$request->newString. ' Created!');
    }

    public function search(SearchFormRequest $request)
    {
        $search_value = $request->search;
        $fields = collect(\DB::getSchemaBuilder()->getColumnListing('strings'));
        $columns = $fields->flip()->except(['code','created_at','updated_at'])->keys();

        $query = Strings::select('*');
        $query->where('en', 'LIKE', '%' . $search_value . '%');

        foreach($columns as $column)
        {
          $query->orWhere($column, 'LIKE', '%' . $search_value . '%');
        }

        $result = $query->get();
        return view('langs::search_result')->with('result', $result)->with('search_value', $search_value);
    }

    public function string($code)
    {
        $string = Strings::find($code);
        return view('langs::lang')->with('string', $string);
    }

    public function publishAll()
    {
        $fields = collect(\DB::getSchemaBuilder()->getColumnListing('strings'));
        $columns = $fields->flip()->except(['code','en','created_at','updated_at'])->keys();

        foreach ($columns as $lang) {
            $list = Strings::pluck($lang,'en');
            $json = json_encode_prettify($list);
            $this->files->put(base_path('lang/'.$lang.'.json'),$json);
        }

        return redirect()->back()->with(config('elseyyid-location.message_success_variable'), __('All Json Files Published!'));
    }
}
