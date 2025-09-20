# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Receipt\MarkCodeInfo
### Namespace: [\YooKassa\Model\Receipt](../namespaces/yookassa-model-receipt.md)
---
**Summary:**

Class MarkCodeInfo.

**Description:**

Код товара (тег в 54 ФЗ — 1163).
Обязательный параметр, если одновременно выполняются эти условия:
* вы используете Чеки от ЮKassa или онлайн-кассу, обновленную до ФФД 1.2;
* товар нужно [маркировать](http://docs.cntd.ru/document/902192509).

Должно быть заполнено хотя бы одно поле.

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [MIN_LENGTH](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#constant_MIN_LENGTH) |  |  |
| public | [MAX_UNKNOWN_LENGTH](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#constant_MAX_UNKNOWN_LENGTH) |  |  |
| public | [EAN_8_LENGTH](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#constant_EAN_8_LENGTH) |  |  |
| public | [EAN_13_LENGTH](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#constant_EAN_13_LENGTH) |  |  |
| public | [ITF_14_LENGTH](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#constant_ITF_14_LENGTH) |  |  |
| public | [MAX_GS_10_LENGTH](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#constant_MAX_GS_10_LENGTH) |  |  |
| public | [MAX_GS_1M_LENGTH](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#constant_MAX_GS_1M_LENGTH) |  |  |
| public | [MAX_SHORT_LENGTH](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#constant_MAX_SHORT_LENGTH) |  |  |
| public | [FUR_LENGTH](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#constant_FUR_LENGTH) |  |  |
| public | [EGAIS_20_LENGTH](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#constant_EGAIS_20_LENGTH) |  |  |
| public | [EGAIS_30_LENGTH](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#constant_EGAIS_30_LENGTH) |  |  |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$ean_13](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#property_ean_13) |  | Код товара в формате EAN-13 (тег в 54 ФЗ — 1302) |
| public | [$ean_8](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#property_ean_8) |  | Код товара в формате EAN-8 (тег в 54 ФЗ — 1301) |
| public | [$egais_20](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#property_egais_20) |  | Код товара в формате ЕГАИС-2.0 (тег в 54 ФЗ — 1308) |
| public | [$egais_30](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#property_egais_30) |  | Код товара в формате ЕГАИС-3.0 (тег в 54 ФЗ — 1309) |
| public | [$fur](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#property_fur) |  | Контрольно-идентификационный знак мехового изделия (тег в 54 ФЗ — 1307) |
| public | [$gs_10](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#property_gs_10) |  | Код товара в формате GS1.0 (тег в 54 ФЗ — 1304). <br/>Можно передавать, если используете онлайн-кассу Orange Data, aQsi, Кит Инвест, АТОЛ Онлайн. |
| public | [$gs_1m](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#property_gs_1m) |  | Код товара в формате GS1.M (тег в 54 ФЗ — 1305) |
| public | [$itf_14](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#property_itf_14) |  | Код товара в формате ITF-14 (тег в 54 ФЗ — 1303) |
| public | [$mark_code_raw](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#property_mark_code_raw) |  | Код товара в том виде, в котором он был прочитан сканером (тег в 54 ФЗ — 2000). <br/>Нужно передавать, если используете онлайн-кассу Orange Data. <br/>Пример: ~`010460406000590021N4N57RTCBUZTQ\\u001d2403054002410161218\\u001d1424010191ffd0\\u001g92tIAF/YVpU4roQS3M/m4z78yFq0nc/WsSmLeX6QkF/YVWwy5IMYAeiQ91Xa2m/fFSJcOkb2N+uUUtfr4n0mOX0Q==` |
| public | [$markCodeRaw](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#property_markCodeRaw) |  | Код товара в том виде, в котором он был прочитан сканером (тег в 54 ФЗ — 2000). <br/>Нужно передавать, если используете онлайн-кассу Orange Data. <br/>Пример: ~`010460406000590021N4N57RTCBUZTQ\\u001d2403054002410161218\\u001d1424010191ffd0\\u001g92tIAF/YVpU4roQS3M/m4z78yFq0nc/WsSmLeX6QkF/YVWwy5IMYAeiQ91Xa2m/fFSJcOkb2N+uUUtfr4n0mOX0Q==` |
| public | [$short](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#property_short) |  | Код товара в формате короткого кода маркировки (тег в 54 ФЗ — 1306) |
| public | [$unknown](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#property_unknown) |  | Нераспознанный код товара (тег в 54 ФЗ — 1300) |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractObject.md#method___construct) |  | AbstractObject constructor. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getEan13()](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#method_getEan13) |  | Возвращает ean_13. |
| public | [getEan8()](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#method_getEan8) |  | Возвращает ean_8. |
| public | [getEgais20()](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#method_getEgais20) |  | Возвращает egais_20. |
| public | [getEgais30()](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#method_getEgais30) |  | Возвращает egais_30. |
| public | [getFur()](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#method_getFur) |  | Возвращает fur. |
| public | [getGs10()](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#method_getGs10) |  | Возвращает gs_10. |
| public | [getGs1m()](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#method_getGs1m) |  | Возвращает gs_1m. |
| public | [getItf14()](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#method_getItf14) |  | Возвращает itf_14. |
| public | [getMarkCodeRaw()](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#method_getMarkCodeRaw) |  | Возвращает исходный код товара. |
| public | [getShort()](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#method_getShort) |  | Возвращает short. |
| public | [getUnknown()](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#method_getUnknown) |  | Возвращает unknown. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setEan13()](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#method_setEan13) |  | Устанавливает ean_13. |
| public | [setEan8()](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#method_setEan8) |  | Устанавливает ean_8. |
| public | [setEgais20()](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#method_setEgais20) |  | Устанавливает egais_20. |
| public | [setEgais30()](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#method_setEgais30) |  | Устанавливает egais_30. |
| public | [setFur()](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#method_setFur) |  | Устанавливает fur. |
| public | [setGs10()](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#method_setGs10) |  | Устанавливает gs_10. |
| public | [setGs1m()](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#method_setGs1m) |  | Устанавливает gs_1m. |
| public | [setItf14()](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#method_setItf14) |  | Устанавливает itf_14. |
| public | [setMarkCodeRaw()](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#method_setMarkCodeRaw) |  | Устанавливает исходный код товара. |
| public | [setShort()](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#method_setShort) |  | Устанавливает short. |
| public | [setUnknown()](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md#method_setUnknown) |  | Устанавливает unknown. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Model/Receipt/MarkCodeInfo.php](../../lib/Model/Receipt/MarkCodeInfo.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Model\Receipt\MarkCodeInfo

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Constants
<a name="constant_MIN_LENGTH" class="anchor"></a>
###### MIN_LENGTH
```php
MIN_LENGTH = 1 : int
```


<a name="constant_MAX_UNKNOWN_LENGTH" class="anchor"></a>
###### MAX_UNKNOWN_LENGTH
```php
MAX_UNKNOWN_LENGTH = 32 : int
```


<a name="constant_EAN_8_LENGTH" class="anchor"></a>
###### EAN_8_LENGTH
```php
EAN_8_LENGTH = 8 : int
```


<a name="constant_EAN_13_LENGTH" class="anchor"></a>
###### EAN_13_LENGTH
```php
EAN_13_LENGTH = 13 : int
```


<a name="constant_ITF_14_LENGTH" class="anchor"></a>
###### ITF_14_LENGTH
```php
ITF_14_LENGTH = 14 : int
```


<a name="constant_MAX_GS_10_LENGTH" class="anchor"></a>
###### MAX_GS_10_LENGTH
```php
MAX_GS_10_LENGTH = 38 : int
```


<a name="constant_MAX_GS_1M_LENGTH" class="anchor"></a>
###### MAX_GS_1M_LENGTH
```php
MAX_GS_1M_LENGTH = 200 : int
```


<a name="constant_MAX_SHORT_LENGTH" class="anchor"></a>
###### MAX_SHORT_LENGTH
```php
MAX_SHORT_LENGTH = 38 : int
```


<a name="constant_FUR_LENGTH" class="anchor"></a>
###### FUR_LENGTH
```php
FUR_LENGTH = 20 : int
```


<a name="constant_EGAIS_20_LENGTH" class="anchor"></a>
###### EGAIS_20_LENGTH
```php
EGAIS_20_LENGTH = 33 : int
```


<a name="constant_EGAIS_30_LENGTH" class="anchor"></a>
###### EGAIS_30_LENGTH
```php
EGAIS_30_LENGTH = 14 : int
```



---
## Properties
<a name="property_ean_13"></a>
#### public $ean_13 : string
---
***Description***

Код товара в формате EAN-13 (тег в 54 ФЗ — 1302)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_ean_8"></a>
#### public $ean_8 : string
---
***Description***

Код товара в формате EAN-8 (тег в 54 ФЗ — 1301)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_egais_20"></a>
#### public $egais_20 : string
---
***Description***

Код товара в формате ЕГАИС-2.0 (тег в 54 ФЗ — 1308)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_egais_30"></a>
#### public $egais_30 : string
---
***Description***

Код товара в формате ЕГАИС-3.0 (тег в 54 ФЗ — 1309)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_fur"></a>
#### public $fur : string
---
***Description***

Контрольно-идентификационный знак мехового изделия (тег в 54 ФЗ — 1307)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_gs_10"></a>
#### public $gs_10 : string
---
***Description***

Код товара в формате GS1.0 (тег в 54 ФЗ — 1304). <br/>Можно передавать, если используете онлайн-кассу Orange Data, aQsi, Кит Инвест, АТОЛ Онлайн.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_gs_1m"></a>
#### public $gs_1m : string
---
***Description***

Код товара в формате GS1.M (тег в 54 ФЗ — 1305)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_itf_14"></a>
#### public $itf_14 : string
---
***Description***

Код товара в формате ITF-14 (тег в 54 ФЗ — 1303)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_mark_code_raw"></a>
#### public $mark_code_raw : string
---
***Description***

Код товара в том виде, в котором он был прочитан сканером (тег в 54 ФЗ — 2000). <br/>Нужно передавать, если используете онлайн-кассу Orange Data. <br/>Пример: ~`010460406000590021N4N57RTCBUZTQ\\u001d2403054002410161218\\u001d1424010191ffd0\\u001g92tIAF/YVpU4roQS3M/m4z78yFq0nc/WsSmLeX6QkF/YVWwy5IMYAeiQ91Xa2m/fFSJcOkb2N+uUUtfr4n0mOX0Q==`

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_markCodeRaw"></a>
#### public $markCodeRaw : string
---
***Description***

Код товара в том виде, в котором он был прочитан сканером (тег в 54 ФЗ — 2000). <br/>Нужно передавать, если используете онлайн-кассу Orange Data. <br/>Пример: ~`010460406000590021N4N57RTCBUZTQ\\u001d2403054002410161218\\u001d1424010191ffd0\\u001g92tIAF/YVpU4roQS3M/m4z78yFq0nc/WsSmLeX6QkF/YVWwy5IMYAeiQ91Xa2m/fFSJcOkb2N+uUUtfr4n0mOX0Q==`

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_short"></a>
#### public $short : string
---
***Description***

Код товара в формате короткого кода маркировки (тег в 54 ФЗ — 1306)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_unknown"></a>
#### public $unknown : string
---
***Description***

Нераспознанный код товара (тег в 54 ФЗ — 1300)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct(array|null $data = []) : mixed
```

**Summary**

AbstractObject constructor.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR null</code> | data  |  |

**Returns:** mixed - 


<a name="method___get" class="anchor"></a>
#### public __get() : mixed

```php
public __get(string $propertyName) : mixed
```

**Summary**

Возвращает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя свойства |

**Returns:** mixed - Значение свойства


<a name="method___isset" class="anchor"></a>
#### public __isset() : bool

```php
public __isset(string $propertyName) : bool
```

**Summary**

Проверяет наличие свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя проверяемого свойства |

**Returns:** bool - True если свойство имеется, false если нет


<a name="method___set" class="anchor"></a>
#### public __set() : void

```php
public __set(string $propertyName, mixed $value) : void
```

**Summary**

Устанавливает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя свойства |
| <code lang="php">mixed</code> | value  | Значение свойства |

**Returns:** void - 


<a name="method___unset" class="anchor"></a>
#### public __unset() : void

```php
public __unset(string $propertyName) : void
```

**Summary**

Удаляет свойство.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя удаляемого свойства |

**Returns:** void - 


<a name="method_fromArray" class="anchor"></a>
#### public fromArray() : void

```php
public fromArray(array|\Traversable $sourceArray) : void
```

**Summary**

Устанавливает значения свойств текущего объекта из массива.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \Traversable</code> | sourceArray  | Ассоциативный массив с настройками |

**Returns:** void - 


<a name="method_getEan13" class="anchor"></a>
#### public getEan13() : string|null

```php
public getEan13() : string|null
```

**Summary**

Возвращает ean_13.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\MarkCodeInfo](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md)

**Returns:** string|null - 


<a name="method_getEan8" class="anchor"></a>
#### public getEan8() : string|null

```php
public getEan8() : string|null
```

**Summary**

Возвращает ean_8.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\MarkCodeInfo](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md)

**Returns:** string|null - 


<a name="method_getEgais20" class="anchor"></a>
#### public getEgais20() : string|null

```php
public getEgais20() : string|null
```

**Summary**

Возвращает egais_20.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\MarkCodeInfo](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md)

**Returns:** string|null - 


<a name="method_getEgais30" class="anchor"></a>
#### public getEgais30() : string|null

```php
public getEgais30() : string|null
```

**Summary**

Возвращает egais_30.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\MarkCodeInfo](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md)

**Returns:** string|null - 


<a name="method_getFur" class="anchor"></a>
#### public getFur() : string|null

```php
public getFur() : string|null
```

**Summary**

Возвращает fur.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\MarkCodeInfo](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md)

**Returns:** string|null - 


<a name="method_getGs10" class="anchor"></a>
#### public getGs10() : string|null

```php
public getGs10() : string|null
```

**Summary**

Возвращает gs_10.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\MarkCodeInfo](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md)

**Returns:** string|null - 


<a name="method_getGs1m" class="anchor"></a>
#### public getGs1m() : string|null

```php
public getGs1m() : string|null
```

**Summary**

Возвращает gs_1m.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\MarkCodeInfo](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md)

**Returns:** string|null - 


<a name="method_getItf14" class="anchor"></a>
#### public getItf14() : string|null

```php
public getItf14() : string|null
```

**Summary**

Возвращает itf_14.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\MarkCodeInfo](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md)

**Returns:** string|null - 


<a name="method_getMarkCodeRaw" class="anchor"></a>
#### public getMarkCodeRaw() : string|null

```php
public getMarkCodeRaw() : string|null
```

**Summary**

Возвращает исходный код товара.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\MarkCodeInfo](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md)

**Returns:** string|null - Исходный код товара


<a name="method_getShort" class="anchor"></a>
#### public getShort() : string|null

```php
public getShort() : string|null
```

**Summary**

Возвращает short.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\MarkCodeInfo](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md)

**Returns:** string|null - 


<a name="method_getUnknown" class="anchor"></a>
#### public getUnknown() : string|null

```php
public getUnknown() : string|null
```

**Summary**

Возвращает unknown.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\MarkCodeInfo](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md)

**Returns:** string|null - 


<a name="method_getValidator" class="anchor"></a>
#### public getValidator() : \YooKassa\Validator\Validator

```php
public getValidator() : \YooKassa\Validator\Validator
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** \YooKassa\Validator\Validator - 


<a name="method_jsonSerialize" class="anchor"></a>
#### public jsonSerialize() : array

```php
public jsonSerialize() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\MarkCodeInfo](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта


<a name="method_offsetExists" class="anchor"></a>
#### public offsetExists() : bool

```php
public offsetExists(string $offset) : bool
```

**Summary**

Проверяет наличие свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя проверяемого свойства |

**Returns:** bool - True если свойство имеется, false если нет


<a name="method_offsetGet" class="anchor"></a>
#### public offsetGet() : mixed

```php
public offsetGet(string $offset) : mixed
```

**Summary**

Возвращает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя свойства |

**Returns:** mixed - Значение свойства


<a name="method_offsetSet" class="anchor"></a>
#### public offsetSet() : void

```php
public offsetSet(string $offset, mixed $value) : void
```

**Summary**

Устанавливает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя свойства |
| <code lang="php">mixed</code> | value  | Значение свойства |

**Returns:** void - 


<a name="method_offsetUnset" class="anchor"></a>
#### public offsetUnset() : void

```php
public offsetUnset(string $offset) : void
```

**Summary**

Удаляет свойство.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя удаляемого свойства |

**Returns:** void - 


<a name="method_setEan13" class="anchor"></a>
#### public setEan13() : self

```php
public setEan13(string|null $ean_13 = null) : self
```

**Summary**

Устанавливает ean_13.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\MarkCodeInfo](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | ean_13  | Код товара в формате EAN-13 (тег в 54 ФЗ — 1302). |

**Returns:** self - 


<a name="method_setEan8" class="anchor"></a>
#### public setEan8() : self

```php
public setEan8(string|null $ean_8 = null) : self
```

**Summary**

Устанавливает ean_8.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\MarkCodeInfo](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | ean_8  | Код товара в формате EAN-8 (тег в 54 ФЗ — 1301). |

**Returns:** self - 


<a name="method_setEgais20" class="anchor"></a>
#### public setEgais20() : self

```php
public setEgais20(string|null $egais_20 = null) : self
```

**Summary**

Устанавливает egais_20.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\MarkCodeInfo](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | egais_20  | Код товара в формате ЕГАИС-2.0 (тег в 54 ФЗ — 1308). |

**Returns:** self - 


<a name="method_setEgais30" class="anchor"></a>
#### public setEgais30() : self

```php
public setEgais30(string|null $egais_30 = null) : self
```

**Summary**

Устанавливает egais_30.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\MarkCodeInfo](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | egais_30  | Код товара в формате ЕГАИС-3.0 (тег в 54 ФЗ — 1309). |

**Returns:** self - 


<a name="method_setFur" class="anchor"></a>
#### public setFur() : self

```php
public setFur(string|null $fur = null) : self
```

**Summary**

Устанавливает fur.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\MarkCodeInfo](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | fur  | Контрольно-идентификационный знак мехового изделия (тег в 54 ФЗ — 1307). |

**Returns:** self - 


<a name="method_setGs10" class="anchor"></a>
#### public setGs10() : self

```php
public setGs10(string|null $gs_10 = null) : self
```

**Summary**

Устанавливает gs_10.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\MarkCodeInfo](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | gs_10  | Код товара в формате GS1.0 (тег в 54 ФЗ — 1304). <br/>Онлайн-кассы, которые поддерживают этот параметр: **Orange Data**, **aQsi**, **Кит Инвест**. |

**Returns:** self - 


<a name="method_setGs1m" class="anchor"></a>
#### public setGs1m() : self

```php
public setGs1m(string|null $gs_1m = null) : self
```

**Summary**

Устанавливает gs_1m.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\MarkCodeInfo](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | gs_1m  | Код товара в формате GS1.M (тег в 54 ФЗ — 1305). |

**Returns:** self - 


<a name="method_setItf14" class="anchor"></a>
#### public setItf14() : self

```php
public setItf14(string|null $itf_14 = null) : self
```

**Summary**

Устанавливает itf_14.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\MarkCodeInfo](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | itf_14  | Код товара в формате ITF-14 (тег в 54 ФЗ — 1303). |

**Returns:** self - 


<a name="method_setMarkCodeRaw" class="anchor"></a>
#### public setMarkCodeRaw() : self

```php
public setMarkCodeRaw(string|null $mark_code_raw = null) : self
```

**Summary**

Устанавливает исходный код товара.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\MarkCodeInfo](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | mark_code_raw  | Исходный код товара |

**Returns:** self - 


<a name="method_setShort" class="anchor"></a>
#### public setShort() : self

```php
public setShort(string|null $short = null) : self
```

**Summary**

Устанавливает short.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\MarkCodeInfo](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | short  | Код товара в формате короткого кода маркировки (тег в 54 ФЗ — 1306). |

**Returns:** self - 


<a name="method_setUnknown" class="anchor"></a>
#### public setUnknown() : self

```php
public setUnknown(string|null $unknown = null) : self
```

**Summary**

Устанавливает unknown.

**Details:**
* Inherited From: [\YooKassa\Model\Receipt\MarkCodeInfo](../classes/YooKassa-Model-Receipt-MarkCodeInfo.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | unknown  | Нераспознанный код товара (тег в 54 ФЗ — 1300). |

**Returns:** self - 


<a name="method_toArray" class="anchor"></a>
#### public toArray() : array

```php
public toArray() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации
Является алиасом метода AbstractObject::jsonSerialize().

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта


<a name="method_getUnknownProperties" class="anchor"></a>
#### protected getUnknownProperties() : array

```php
protected getUnknownProperties() : array
```

**Summary**

Возвращает массив свойств которые не существуют, но были заданы у объекта.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив с не существующими у текущего объекта свойствами


<a name="method_validatePropertyValue" class="anchor"></a>
#### protected validatePropertyValue() : mixed

```php
protected validatePropertyValue(string $propertyName, mixed $propertyValue) : mixed
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  |  |
| <code lang="php">mixed</code> | propertyValue  |  |

**Returns:** mixed - 



---

### Top Namespaces

* [\YooKassa](../namespaces/yookassa.md)

---

### Reports
* [Errors - 0](../reports/errors.md)
* [Markers - 0](../reports/markers.md)
* [Deprecated - 40](../reports/deprecated.md)

---

This document was automatically generated from source code comments on 2025-09-04 using [phpDocumentor](http://www.phpdoc.org/)

&copy; 2025 YooMoney