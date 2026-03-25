<?php
return ['name'=>'Treasury','features'=>['ai'=>env('TREASURY_AI_ENABLED',true),'api'=>env('TREASURY_API_ENABLED',true),'payments'=>true,'reconciliation'=>true,'forecast'=>true],];