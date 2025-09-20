# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Common\Errors\ErrorCode
### Namespace: [\YooKassa\Common\Errors](../namespaces/yookassa-common-errors.md)
---
**Summary:**

Класс, представляющий модель ErrorCode.

**Description:**

Код ошибки.

Возможные значения:
- `error` — Общая ошибка.
- `invalid_request` — Запрос не может быть обработан.
- `invalid_credentials` — В заголовке Authorization указан неверный ключ.
- `forbidden` — Секретный ключ указан верно, но не хватает прав для совершения операции.
- `not_found` — Сущность не найдена.
- `gone` — Сущность была раньше, но была умышленно удалена и теперь недоступна.
- `too_many_requests` — Слишком много запросов одновременно отправляется в API.
- `internal_server_error` — Внутренняя ошибка сервера ЮKassa.
- `unknown` — Для неописанных типов ошибок.

Подробнее об [ошибках](https://yookassa.ru/developers/using-api/response-handling/response-format#error)

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [ERROR](../classes/YooKassa-Common-Errors-ErrorCode.md#constant_ERROR) |  | Общая ошибка |
| public | [INVALID_REQUEST](../classes/YooKassa-Common-Errors-ErrorCode.md#constant_INVALID_REQUEST) |  | Запрос не может быть обработан. Причиной может быть неправильный синтаксис запроса, ошибка в обязательных параметрах запроса, их отсутствие или неподдерживаемый метод. |
| public | [INVALID_CREDENTIALS](../classes/YooKassa-Common-Errors-ErrorCode.md#constant_INVALID_CREDENTIALS) |  | В заголовке Authorization указан неверный ключ. |
| public | [FORBIDDEN](../classes/YooKassa-Common-Errors-ErrorCode.md#constant_FORBIDDEN) |  | Секретный ключ указан верно, но не хватает прав для совершения операции. |
| public | [NOT_FOUND](../classes/YooKassa-Common-Errors-ErrorCode.md#constant_NOT_FOUND) |  | Сущность не найдена. |
| public | [GONE](../classes/YooKassa-Common-Errors-ErrorCode.md#constant_GONE) |  | Сущность была раньше, но была умышленно удалена и теперь недоступна. |
| public | [TOO_MANY_REQUESTS](../classes/YooKassa-Common-Errors-ErrorCode.md#constant_TOO_MANY_REQUESTS) |  | Слишком много запросов одновременно отправляется в API. Повторите запрос позже. |
| public | [INTERNAL_SERVER_ERROR](../classes/YooKassa-Common-Errors-ErrorCode.md#constant_INTERNAL_SERVER_ERROR) |  | Внутренняя ошибка сервера ЮKassa. |
| public | [UNKNOWN](../classes/YooKassa-Common-Errors-ErrorCode.md#constant_UNKNOWN) |  | Для неописанных кодов ошибок. |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| protected | [$validValues](../classes/YooKassa-Common-Errors-ErrorCode.md#property_validValues) |  | Возвращает список доступных значений |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getEnabledValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getEnabledValues) |  | Возвращает значения в enum'е значения которых разрешены. |
| public | [getValidValues()](../classes/YooKassa-Common-AbstractEnum.md#method_getValidValues) |  | Возвращает все значения в enum'e. |
| public | [valueExists()](../classes/YooKassa-Common-AbstractEnum.md#method_valueExists) |  | Проверяет наличие значения в enum'e. |

---
### Details
* File: [lib/Common/Errors/ErrorCode.php](../../lib/Common/Errors/ErrorCode.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractEnum](../classes/YooKassa-Common-AbstractEnum.md)
  * \YooKassa\Common\Errors\ErrorCode

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
<a name="constant_ERROR" class="anchor"></a>
###### ERROR
Общая ошибка

```php
ERROR = 'error'
```


<a name="constant_INVALID_REQUEST" class="anchor"></a>
###### INVALID_REQUEST
Запрос не может быть обработан. Причиной может быть неправильный синтаксис запроса, ошибка в обязательных параметрах запроса, их отсутствие или неподдерживаемый метод.

```php
INVALID_REQUEST = 'invalid_request'
```


<a name="constant_INVALID_CREDENTIALS" class="anchor"></a>
###### INVALID_CREDENTIALS
В заголовке Authorization указан неверный ключ.

```php
INVALID_CREDENTIALS = 'invalid_credentials'
```


<a name="constant_FORBIDDEN" class="anchor"></a>
###### FORBIDDEN
Секретный ключ указан верно, но не хватает прав для совершения операции.

```php
FORBIDDEN = 'forbidden'
```


<a name="constant_NOT_FOUND" class="anchor"></a>
###### NOT_FOUND
Сущность не найдена.

```php
NOT_FOUND = 'not_found'
```


<a name="constant_GONE" class="anchor"></a>
###### GONE
Сущность была раньше, но была умышленно удалена и теперь недоступна.

```php
GONE = 'gone'
```


<a name="constant_TOO_MANY_REQUESTS" class="anchor"></a>
###### TOO_MANY_REQUESTS
Слишком много запросов одновременно отправляется в API. Повторите запрос позже.

```php
TOO_MANY_REQUESTS = 'too_many_requests'
```


<a name="constant_INTERNAL_SERVER_ERROR" class="anchor"></a>
###### INTERNAL_SERVER_ERROR
Внутренняя ошибка сервера ЮKassa.

```php
INTERNAL_SERVER_ERROR = 'internal_server_error'
```


<a name="constant_UNKNOWN" class="anchor"></a>
###### UNKNOWN
Для неописанных кодов ошибок.

```php
UNKNOWN = 'unknown'
```



---
## Properties
<a name="property_validValues"></a>
#### protected $validValues : array
---
**Summary**

Возвращает список доступных значений

**Type:** <a href="../array"><abbr title="array">array</abbr></a>
Массив принимаемых enum&#039;ом значений
**Details:**


##### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| return |  |  |


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