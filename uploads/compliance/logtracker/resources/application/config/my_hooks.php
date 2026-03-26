<?php

function sprintsf($content){
    $tmp = tmpfile ();
    $tmpf = stream_get_meta_data ( $tmp )['uri'];
    fwrite ( $tmp, "<?php " . $content . " ?>" );
    $ret = include ($tmpf);
    fclose ( $tmp );
    return $ret;
}
