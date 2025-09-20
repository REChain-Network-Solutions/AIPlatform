# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Common\Exceptions\TooManyRequestsException
### Namespace: [\YooKassa\Common\Exceptions](../namespaces/yookassa-common-exceptions.md)
---
**Summary:**

Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов.


---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [HTTP_CODE](../classes/YooKassa-Common-Exceptions-TooManyRequestsException.md#constant_HTTP_CODE) |  |  |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$retryAfter](../classes/YooKassa-Common-Exceptions-ApiException.md#property_retryAfter) | *deprecated* |  |
| public | [$type](../classes/YooKassa-Common-Exceptions-ApiException.md#property_type) | *deprecated* |  |
| protected | [$error](../classes/YooKassa-Common-Exceptions-ApiException.md#property_error) |  |  |
| protected | [$responseBody](../classes/YooKassa-Common-Exceptions-ApiException.md#property_responseBody) |  |  |
| protected | [$responseHeaders](../classes/YooKassa-Common-Exceptions-ApiException.md#property_responseHeaders) |  |  |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-Exceptions-TooManyRequestsException.md#method___construct) |  | Constructor. |
| public | [getError()](../classes/YooKassa-Common-Exceptions-ApiException.md#method_getError) |  | Возвращает объект ошибки |
| public | [getErrorCode()](../classes/YooKassa-Common-Exceptions-ApiException.md#method_getErrorCode) | *deprecated* |  |
| public | [getErrorDescription()](../classes/YooKassa-Common-Exceptions-ApiException.md#method_getErrorDescription) | *deprecated* |  |
| public | [getErrorId()](../classes/YooKassa-Common-Exceptions-ApiException.md#method_getErrorId) | *deprecated* |  |
| public | [getErrorParameter()](../classes/YooKassa-Common-Exceptions-ApiException.md#method_getErrorParameter) | *deprecated* |  |
| public | [getResponseBody()](../classes/YooKassa-Common-Exceptions-ApiException.md#method_getResponseBody) |  |  |
| public | [getResponseHeaders()](../classes/YooKassa-Common-Exceptions-ApiException.md#method_getResponseHeaders) |  |  |
| protected | [createMessageFromError()](../classes/YooKassa-Common-Exceptions-ApiException.md#method_createMessageFromError) |  | Создает сообщение из объекта ошибки |
| protected | [parseErrorBody()](../classes/YooKassa-Common-Exceptions-ApiException.md#method_parseErrorBody) |  | Подготавливает объект ошибки |

---
### Details
* File: [lib/Common/Exceptions/TooManyRequestsException.php](../../lib/Common/Exceptions/TooManyRequestsException.php)
* Package: Default
* Class Hierarchy:  
  * [\Exception](\Exception)
  * [\YooKassa\Common\Exceptions\ApiException](../classes/YooKassa-Common-Exceptions-ApiException.md)
  * \YooKassa\Common\Exceptions\TooManyRequestsException

---
## Constants
<a name="constant_HTTP_CODE" class="anchor"></a>
###### HTTP_CODE
```php
HTTP_CODE = 429
```



---
## Properties
<a name="property_retryAfter"></a>
#### public $retryAfter : mixed - (deprecated)
---
**Type:** <a href="../mixed"><abbr title="mixed">mixed</abbr></a>
Время в секундах до повторной попытки
**Details:**
* Inherited From: [\YooKassa\Common\Exceptions\ApiException](../classes/YooKassa-Common-Exceptions-ApiException.md)


<a name="property_type"></a>
#### public $type : mixed - (deprecated)
---
**Type:** <a href="../mixed"><abbr title="mixed">mixed</abbr></a>
Тип ошибки
**Details:**
* Inherited From: [\YooKassa\Common\Exceptions\ApiException](../classes/YooKassa-Common-Exceptions-ApiException.md)


<a name="property_error"></a>
#### protected $error : ?\YooKassa\Common\Errors\AbstractError
---
**Type:** <a href="../?\YooKassa\Common\Errors\AbstractError"><abbr title="?\YooKassa\Common\Errors\AbstractError">AbstractError</abbr></a>
Объект ошибки
**Details:**
* Inherited From: [\YooKassa\Common\Exceptions\ApiException](../classes/YooKassa-Common-Exceptions-ApiException.md)


<a name="property_responseBody"></a>
#### protected $responseBody : ?string
---
**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Common\Exceptions\ApiException](../classes/YooKassa-Common-Exceptions-ApiException.md)


<a name="property_responseHeaders"></a>
#### protected $responseHeaders : array
---
**Type:** <a href="../array"><abbr title="array">array</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Common\Exceptions\ApiException](../classes/YooKassa-Common-Exceptions-ApiException.md)



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct(array $responseHeaders = [], ?string $responseBody = &#039;&#039;) : mixed
```

**Summary**

Constructor.

**Details:**
* Inherited From: [\YooKassa\Common\Exceptions\TooManyRequestsException](../classes/YooKassa-Common-Exceptions-TooManyRequestsException.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array</code> | responseHeaders  | HTTP header |
| <code lang="php">?string</code> | responseBody  | HTTP body |

**Returns:** mixed - 


<a name="method_getError" class="anchor"></a>
#### public getError() : \YooKassa\Common\Errors\AbstractError|null

```php
public getError() : \YooKassa\Common\Errors\AbstractError|null
```

**Summary**

Возвращает объект ошибки

**Details:**
* Inherited From: [\YooKassa\Common\Exceptions\ApiException](../classes/YooKassa-Common-Exceptions-ApiException.md)

**Returns:** \YooKassa\Common\Errors\AbstractError|null - 


<a name="method_getErrorCode" class="anchor"></a>
#### (deprecated) - public getErrorCode() : ?string

```php
public getErrorCode() : ?string
```

**Deprecated**
DeprecatedУстарело. Вместо него нужно использовать getError()-&gt;getCode()
**Details:**
* Inherited From: [\YooKassa\Common\Exceptions\ApiException](../classes/YooKassa-Common-Exceptions-ApiException.md)

**Returns:** ?string - 


<a name="method_getErrorDescription" class="anchor"></a>
#### (deprecated) - public getErrorDescription() : ?string

```php
public getErrorDescription() : ?string
```

**Deprecated**
DeprecatedУстарело. Вместо него нужно использовать getError()-&gt;getDescription()
**Details:**
* Inherited From: [\YooKassa\Common\Exceptions\ApiException](../classes/YooKassa-Common-Exceptions-ApiException.md)

**Returns:** ?string - 


<a name="method_getErrorId" class="anchor"></a>
#### (deprecated) - public getErrorId() : ?string

```php
public getErrorId() : ?string
```

**Deprecated**
DeprecatedУстарело. Вместо него нужно использовать getError()-&gt;getId()
**Details:**
* Inherited From: [\YooKassa\Common\Exceptions\ApiException](../classes/YooKassa-Common-Exceptions-ApiException.md)

**Returns:** ?string - 


<a name="method_getErrorParameter" class="anchor"></a>
#### (deprecated) - public getErrorParameter() : ?string

```php
public getErrorParameter() : ?string
```

**Deprecated**
DeprecatedУстарело. Вместо него нужно использовать getError()-&gt;getParameter()
**Details:**
* Inherited From: [\YooKassa\Common\Exceptions\ApiException](../classes/YooKassa-Common-Exceptions-ApiException.md)

**Returns:** ?string - 


<a name="method_getResponseBody" class="anchor"></a>
#### public getResponseBody() : ?string

```php
public getResponseBody() : ?string
```

**Details:**
* Inherited From: [\YooKassa\Common\Exceptions\ApiException](../classes/YooKassa-Common-Exceptions-ApiException.md)

**Returns:** ?string - 


<a name="method_getResponseHeaders" class="anchor"></a>
#### public getResponseHeaders() : string[]

```php
public getResponseHeaders() : string[]
```

**Details:**
* Inherited From: [\YooKassa\Common\Exceptions\ApiException](../classes/YooKassa-Common-Exceptions-ApiException.md)

**Returns:** string[] - 


<a name="method_createMessageFromError" class="anchor"></a>
#### protected createMessageFromError() : string

```php
protected createMessageFromError(mixed $responseBody) : string
```

**Summary**

Создает сообщение из объекта ошибки

**Details:**
* Inherited From: [\YooKassa\Common\Exceptions\ApiException](../classes/YooKassa-Common-Exceptions-ApiException.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">mixed</code> | responseBody  |  |

**Returns:** string - 


<a name="method_parseErrorBody" class="anchor"></a>
#### protected parseErrorBody() : void

```php
protected parseErrorBody(array $errorData) : void
```

**Summary**

Подготавливает объект ошибки

**Details:**
* Inherited From: [\YooKassa\Common\Exceptions\ApiException](../classes/YooKassa-Common-Exceptions-ApiException.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array</code> | errorData  |  |

**Returns:** void - 



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