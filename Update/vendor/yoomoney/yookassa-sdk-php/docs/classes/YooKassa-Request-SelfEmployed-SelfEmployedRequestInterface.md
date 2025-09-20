# [YooKassa API SDK](../home.md)

# Interface: SelfEmployedRequestInterface
### Namespace: [\YooKassa\Request\SelfEmployed](../namespaces/yookassa-request-selfemployed.md)
---
**Summary:**

Interface SelfEmployedRequestInterface.

**Description:**

Запрос на создание объекта самозанятого.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getConfirmation()](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestInterface.md#method_getConfirmation) |  | Возвращает сценарий подтверждения. |
| public | [getItn()](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestInterface.md#method_getItn) |  | Возвращает ИНН самозанятого. |
| public | [getPhone()](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestInterface.md#method_getPhone) |  | Возвращает телефон самозанятого. |
| public | [hasConfirmation()](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestInterface.md#method_hasConfirmation) |  | Проверяет наличие сценария подтверждения самозанятого в запросе. |
| public | [hasItn()](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestInterface.md#method_hasItn) |  | Проверяет наличие ИНН самозанятого в запросе. |
| public | [hasPhone()](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestInterface.md#method_hasPhone) |  | Проверяет наличие телефона самозанятого в запросе. |
| public | [setConfirmation()](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestInterface.md#method_setConfirmation) |  | Устанавливает сценарий подтверждения. |
| public | [setItn()](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestInterface.md#method_setItn) |  | Устанавливает ИНН самозанятого. |
| public | [setPhone()](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestInterface.md#method_setPhone) |  | Устанавливает телефон самозанятого. |
| public | [validate()](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestInterface.md#method_validate) |  | Проверяет на валидность текущий объект |

---
### Details
* File: [lib/Request/SelfEmployed/SelfEmployedRequestInterface.php](../../lib/Request/SelfEmployed/SelfEmployedRequestInterface.php)
* Package: \YooKassa\Request
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Interface |
| author |  | cms@yoomoney.ru |
| property |  | ИНН самозанятого. |
| property |  | Телефон самозанятого, который привязан к личному кабинету в сервисе Мой налог. |
| property |  | Сценарий подтверждения пользователем заявки ЮMoney на получение прав для регистрации чеков в сервисе Мой налог. |

---
## Methods
<a name="method_getItn" class="anchor"></a>
#### public getItn() : null|string

```php
public getItn() : null|string
```

**Summary**

Возвращает ИНН самозанятого.

**Details:**
* Inherited From: [\YooKassa\Request\SelfEmployed\SelfEmployedRequestInterface](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestInterface.md)

**Returns:** null|string - ИНН самозанятого


<a name="method_setItn" class="anchor"></a>
#### public setItn() : $this

```php
public setItn(null|string $itn = null) : $this
```

**Summary**

Устанавливает ИНН самозанятого.

**Details:**
* Inherited From: [\YooKassa\Request\SelfEmployed\SelfEmployedRequestInterface](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR string</code> | itn  | ИНН самозанятого |

**Returns:** $this - 


<a name="method_hasItn" class="anchor"></a>
#### public hasItn() : bool

```php
public hasItn() : bool
```

**Summary**

Проверяет наличие ИНН самозанятого в запросе.

**Details:**
* Inherited From: [\YooKassa\Request\SelfEmployed\SelfEmployedRequestInterface](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestInterface.md)

**Returns:** bool - True если ИНН самозанятого задан, false если нет


<a name="method_getPhone" class="anchor"></a>
#### public getPhone() : null|string

```php
public getPhone() : null|string
```

**Summary**

Возвращает телефон самозанятого.

**Details:**
* Inherited From: [\YooKassa\Request\SelfEmployed\SelfEmployedRequestInterface](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestInterface.md)

**Returns:** null|string - Телефон самозанятого


<a name="method_setPhone" class="anchor"></a>
#### public setPhone() : $this

```php
public setPhone(null|string $phone = null) : $this
```

**Summary**

Устанавливает телефон самозанятого.

**Details:**
* Inherited From: [\YooKassa\Request\SelfEmployed\SelfEmployedRequestInterface](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR string</code> | phone  | Телефон самозанятого |

**Returns:** $this - 


<a name="method_hasPhone" class="anchor"></a>
#### public hasPhone() : bool

```php
public hasPhone() : bool
```

**Summary**

Проверяет наличие телефона самозанятого в запросе.

**Details:**
* Inherited From: [\YooKassa\Request\SelfEmployed\SelfEmployedRequestInterface](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestInterface.md)

**Returns:** bool - True если телефон самозанятого задан, false если нет


<a name="method_getConfirmation" class="anchor"></a>
#### public getConfirmation() : ?\YooKassa\Request\SelfEmployed\SelfEmployedRequestConfirmation

```php
public getConfirmation() : ?\YooKassa\Request\SelfEmployed\SelfEmployedRequestConfirmation
```

**Summary**

Возвращает сценарий подтверждения.

**Details:**
* Inherited From: [\YooKassa\Request\SelfEmployed\SelfEmployedRequestInterface](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestInterface.md)

**Returns:** ?\YooKassa\Request\SelfEmployed\SelfEmployedRequestConfirmation - 


<a name="method_setConfirmation" class="anchor"></a>
#### public setConfirmation() : $this

```php
public setConfirmation(null|array|\YooKassa\Model\SelfEmployed\SelfEmployedConfirmation $confirmation = null) : $this
```

**Summary**

Устанавливает сценарий подтверждения.

**Details:**
* Inherited From: [\YooKassa\Request\SelfEmployed\SelfEmployedRequestInterface](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Model\SelfEmployed\SelfEmployedConfirmation</code> | confirmation  | Сценарий подтверждения |

**Returns:** $this - 


<a name="method_hasConfirmation" class="anchor"></a>
#### public hasConfirmation() : bool

```php
public hasConfirmation() : bool
```

**Summary**

Проверяет наличие сценария подтверждения самозанятого в запросе.

**Details:**
* Inherited From: [\YooKassa\Request\SelfEmployed\SelfEmployedRequestInterface](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestInterface.md)

**Returns:** bool - True если сценарий подтверждения самозанятого задан, false если нет


<a name="method_validate" class="anchor"></a>
#### public validate() : bool

```php
public validate() : bool
```

**Summary**

Проверяет на валидность текущий объект

**Details:**
* Inherited From: [\YooKassa\Request\SelfEmployed\SelfEmployedRequestInterface](../classes/YooKassa-Request-SelfEmployed-SelfEmployedRequestInterface.md)

**Returns:** bool - True если объект запроса валиден, false если нет




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