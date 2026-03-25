<?php
namespace Modules\WorkOrders\Database\Seeders;
use Illuminate\Database\Seeder; use Illuminate\Support\Facades\DB;
class FsmIndustryDefaultsSeeder extends Seeder{
  public function run(): void {
    $def = config('fsm_industries.default','general_trades');
    if (DB::getSchemaBuilder()->hasTable('fsm_settings')){
      $row = DB::table('fsm_settings')->orderBy('id','desc')->first();
      if(!$row){
        DB::table('fsm_settings')->insert(['vertical'=>$def,'branding'=>json_encode([]),'features'=>json_encode([]),'created_at'=>now(),'updated_at'=>now()]);
      } else if(empty($row->vertical)){
        DB::table('fsm_settings')->where('id',$row->id)->update(['vertical'=>$def,'updated_at'=>now()]);
      }
    }
  }
}
