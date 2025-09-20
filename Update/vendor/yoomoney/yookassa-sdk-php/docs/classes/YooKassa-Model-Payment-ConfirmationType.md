# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Payment\ConfirmationType
### Namespace: [\YooKassa\Model\Payment](../namespaces/yookassa-model-payment.md)
---
**Summary:**

Класс, представляющий модель ConfirmationType.

**Description:**

Тип пользовательского процесса подтверждения платежа.

Возможные значения:
- `redirect` - Необходимо направить плательщика на страницу партнера
- `external` - Для подтверждения платежа пользователю необходимо совершить действия во внешней системе (например, ответить на смс)
- `code_verification` - Необходимо получить одноразовый код от плательщика для подтверждения платежа
- `embedded` - Необходимо получить токен для checkout.js
- `qr` - Необходимо получить QR-код
- `mobile_application` - Необходимо совершить действия в мобильном приложении

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [REDIRECT](../classes/YooKassa-Model-Payment-ConfirmationType.md#constant_REDIRECT) |  | Необходимо направить плательщика на страницу партнера |
| public | [EXTERNAL](../classes/YooKassa-Model-Payment-ConfirmationType.md#constant_EXTERNAL) |  | Для подтверждения платежа пользователю необходимо совершить действия во внешней системе (например, ответить на смс) |
| public | [CODE_VERIFICATION](../classes/YooKassa-Model-Payment-ConfirmationType.md#constant_CODE_VERIFICATION) | *deprecated* | Необходимо ждать пока плательщик самостоятельно подтвердит платеж. |
| public | [EMBEDDED](../classes/YooKassa-Model-Payment-ConfirmationType.md#constant_EMBEDDED) |  | Необходимо получить одноразовый код от плательщика для подтверждения платежа |
| public | [QR](../classes/YooKassa-Model-Payment-ConfirmationType.md#constant_QR) |  | Необходимо получить QR-код |
| public | [MOBILE_APPLICATION](../classes/YooKassa-Model-Payment-ConfirmationType.md#constant_MOBILE_APPLICATION) |  | Необходимо совершить действия в мобильном приложении |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| protected | [$validValues](../classes/YooKassa-Model-Payment-ConfirmationType.md#property_validValues) |  |  |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getEnabledValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getEnabledValues) |  | Возвращает значения в enum'е значения которых разрешены. |
| public | [getValidValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getValidValues) |  | Возвращает все значения в enum'e. |
| public | [valueExists()](../classes/YooKassa-Common-AbstractEnum.md#method_valueExists) |  | Проверяет наличие значения в enum'e. |

---
### Details
* File: [lib/Model/Payment/ConfirmationType.php](../../lib/Model/Payment/ConfirmationType.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)
  * \YooKassa\Model\Payment\ConfirmationType

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
<a name="constant_REDIRECT" class="anchor"></a>
###### REDIRECT
Необходимо направить плательщика на страницу партнера

```php
REDIRECT = 'redirect'
```


<a name="constant_EXTERNAL" class="anchor"></a>
###### EXTERNAL
Для подтверждения платежа пользователю необходимо совершить действия во внешней системе (например, ответить на смс)

```php
EXTERNAL = 'external'
```


<a name="constant_CODE_VERIFICATION" class="anchor"></a>
###### ~~CODE_VERIFICATION~~
Необходимо ждать пока плательщик самостоятельно подтвердит платеж.

```php
CODE_VERIFICATION = 'code_verification'
```

**deprecated**
Будет удален в следующих версиях

<a name="constant_EMBEDDED" class="anchor"></a>
###### EMBEDDED
Необходимо получить одноразовый код от плательщика для подтверждения платежа

```php
EMBEDDED = 'embedded'
```


<a name="constant_QR" class="anchor"></a>
###### QR
Необходимо получить QR-код

```php
QR = 'qr'
```


<a name="constant_MOBILE_APPLICATION" class="anchor"></a>
###### MOBILE_APPLICATION
Необходимо совершить действия в мобильном приложении

```php
MOBILE_APPLICATION = 'mobile_application'
```



---
## Properties
<a name="property_validValues"></a>
#### protected $validValues : array
---
**Type:** <a href="../array"><abbr title="array">array</abbr></a>
Массив принимаемых enum&#039;ом значений
**Details:**



---
## Methods
<a name="method_getEnabledValues" class="anchor"></a>
#### public getEnabledValues() : string[]

```php
Static public getEnabledValues() : string[]
```

**Summary**

Возвращает значения в enum'е значения которых разрешены.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)

**Returns:** string[] - Массив разрешённых значений


<a name="method_getValidValues" class="anchor"></a>
#### public getValidValues() : array

```php
Static public getValidValues() : array
```

**Summary**

Возвращает все значения в enum'e.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)

**Returns:** array - Массив значений в перечислении


<a name="method_valueExists" class="anchor"></a>
#### public valueExists() : bool

```php
Static public valueExists(mixed $value) : bool
```

**Summary**

Проверяет наличие значения в enum'e.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">mixed</code> | value  | Проверяемое значение |

**Returns:** bool - True если значение имеется, false если нет



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