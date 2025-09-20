# [YooKassa API SDK](../home.md)

# Interface: SelfEmployedInterface
### Namespace: [\YooKassa\Model\SelfEmployed](../namespaces/yookassa-model-selfemployed.md)
---
**Summary:**

Interface SelfEmployedInterface.

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
* [public MIN_LENGTH_ID](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md#constant_MIN_LENGTH_ID)
* [public MAX_LENGTH_ID](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md#constant_MAX_LENGTH_ID)

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [getConfirmation()](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md#method_getConfirmation) |  | Возвращает сценарий подтверждения. |
| public | [getCreatedAt()](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md#method_getCreatedAt) |  | Возвращает время создания объекта самозанятого. |
| public | [getId()](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md#method_getId) |  | Возвращает идентификатор самозанятого. |
| public | [getItn()](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md#method_getItn) |  | Возвращает ИНН самозанятого. |
| public | [getPhone()](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md#method_getPhone) |  | Возвращает телефон самозанятого. |
| public | [getStatus()](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md#method_getStatus) |  | Возвращает статус самозанятого. |
| public | [getTest()](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md#method_getTest) |  | Возвращает признак тестовой операции. |
| public | [setConfirmation()](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md#method_setConfirmation) |  | Устанавливает сценарий подтверждения. |
| public | [setCreatedAt()](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md#method_setCreatedAt) |  | Устанавливает время создания объекта самозанятого. |
| public | [setId()](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md#method_setId) |  | Устанавливает идентификатор самозанятого. |
| public | [setItn()](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md#method_setItn) |  | Устанавливает ИНН самозанятого. |
| public | [setPhone()](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md#method_setPhone) |  | Устанавливает телефон самозанятого. |
| public | [setStatus()](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md#method_setStatus) |  | Устанавливает статус самозанятого. |
| public | [setTest()](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md#method_setTest) |  | Устанавливает признак тестовой операции. |

---
### Details
* File: [lib/Model/SelfEmployed/SelfEmployedInterface.php](../../lib/Model/SelfEmployed/SelfEmployedInterface.php)
* Package: \YooKassa\Model
* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Interface |
| author |  | cms@yoomoney.ru |
| property |  | Идентификатор самозанятого в ЮKassa. |
| property |  | Статус подключения самозанятого и выдачи ЮMoney прав на регистрацию чеков. |
| property |  | Время создания объекта самозанятого. |
| property |  | Время создания объекта самозанятого. |
| property |  | ИНН самозанятого. |
| property |  | Телефон самозанятого, который привязан к личному кабинету в сервисе Мой налог. |
| property |  | Сценарий подтверждения пользователем заявки ЮMoney на получение прав для регистрации чеков в сервисе Мой налог. |
| property |  | Идентификатор самозанятого в ЮKassa. |

---
## Constants
<a name="constant_MIN_LENGTH_ID" class="anchor"></a>
###### MIN_LENGTH_ID
```php
MIN_LENGTH_ID = 36 : int
```


<a name="constant_MAX_LENGTH_ID" class="anchor"></a>
###### MAX_LENGTH_ID
```php
MAX_LENGTH_ID = 50 : int
```



---
## Methods
<a name="method_getId" class="anchor"></a>
#### public getId() : string|null

```php
public getId() : string|null
```

**Summary**

Возвращает идентификатор самозанятого.

**Details:**
* Inherited From: [\YooKassa\Model\SelfEmployed\SelfEmployedInterface](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md)

**Returns:** string|null - Идентификатор самозанятого


<a name="method_setId" class="anchor"></a>
#### public setId() : self

```php
public setId(string $id) : self
```

**Summary**

Устанавливает идентификатор самозанятого.

**Details:**
* Inherited From: [\YooKassa\Model\SelfEmployed\SelfEmployedInterface](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | id  | Идентификатор самозанятого в ЮKassa |

**Returns:** self - 


<a name="method_getStatus" class="anchor"></a>
#### public getStatus() : string|null

```php
public getStatus() : string|null
```

**Summary**

Возвращает статус самозанятого.

**Details:**
* Inherited From: [\YooKassa\Model\SelfEmployed\SelfEmployedInterface](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md)

**Returns:** string|null - Статус самозанятого


<a name="method_setStatus" class="anchor"></a>
#### public setStatus() : $this

```php
public setStatus(string $status) : $this
```

**Summary**

Устанавливает статус самозанятого.

**Details:**
* Inherited From: [\YooKassa\Model\SelfEmployed\SelfEmployedInterface](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | status  | Статус самозанятого |

**Returns:** $this - 


<a name="method_getCreatedAt" class="anchor"></a>
#### public getCreatedAt() : \DateTime|null

```php
public getCreatedAt() : \DateTime|null
```

**Summary**

Возвращает время создания объекта самозанятого.

**Details:**
* Inherited From: [\YooKassa\Model\SelfEmployed\SelfEmployedInterface](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md)

**Returns:** \DateTime|null - Время создания объекта самозанятого


<a name="method_setCreatedAt" class="anchor"></a>
#### public setCreatedAt() : $this

```php
public setCreatedAt(\DateTime|string|null $created_at) : $this
```

**Summary**

Устанавливает время создания объекта самозанятого.

**Details:**
* Inherited From: [\YooKassa\Model\SelfEmployed\SelfEmployedInterface](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\DateTime OR string OR null</code> | created_at  | Время создания объекта самозанятого |

**Returns:** $this - 


<a name="method_getItn" class="anchor"></a>
#### public getItn() : null|string

```php
public getItn() : null|string
```

**Summary**

Возвращает ИНН самозанятого.

**Details:**
* Inherited From: [\YooKassa\Model\SelfEmployed\SelfEmployedInterface](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md)

**Returns:** null|string - ИНН самозанятого


<a name="method_setItn" class="anchor"></a>
#### public setItn() : self

```php
public setItn(null|string $itn = null) : self
```

**Summary**

Устанавливает ИНН самозанятого.

**Details:**
* Inherited From: [\YooKassa\Model\SelfEmployed\SelfEmployedInterface](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR string</code> | itn  | ИНН самозанятого |

**Returns:** self - 


<a name="method_getPhone" class="anchor"></a>
#### public getPhone() : null|string

```php
public getPhone() : null|string
```

**Summary**

Возвращает телефон самозанятого.

**Details:**
* Inherited From: [\YooKassa\Model\SelfEmployed\SelfEmployedInterface](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md)

**Returns:** null|string - Телефон самозанятого


<a name="method_setPhone" class="anchor"></a>
#### public setPhone() : self

```php
public setPhone(null|string $phone = null) : self
```

**Summary**

Устанавливает телефон самозанятого.

**Details:**
* Inherited From: [\YooKassa\Model\SelfEmployed\SelfEmployedInterface](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR string</code> | phone  | Телефон самозанятого |

**Returns:** self - 


<a name="method_getConfirmation" class="anchor"></a>
#### public getConfirmation() : ?\YooKassa\Model\SelfEmployed\SelfEmployedConfirmation

```php
public getConfirmation() : ?\YooKassa\Model\SelfEmployed\SelfEmployedConfirmation
```

**Summary**

Возвращает сценарий подтверждения.

**Details:**
* Inherited From: [\YooKassa\Model\SelfEmployed\SelfEmployedInterface](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md)

**Returns:** ?\YooKassa\Model\SelfEmployed\SelfEmployedConfirmation - 


<a name="method_setConfirmation" class="anchor"></a>
#### public setConfirmation() : self

```php
public setConfirmation(array|\YooKassa\Model\SelfEmployed\SelfEmployedConfirmation|null $confirmation = null) : self
```

**Summary**

Устанавливает сценарий подтверждения.

**Details:**
* Inherited From: [\YooKassa\Model\SelfEmployed\SelfEmployedInterface](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\SelfEmployed\SelfEmployedConfirmation OR null</code> | confirmation  | Сценарий подтверждения |

**Returns:** self - 


<a name="method_getTest" class="anchor"></a>
#### public getTest() : bool

```php
public getTest() : bool
```

**Summary**

Возвращает признак тестовой операции.

**Details:**
* Inherited From: [\YooKassa\Model\SelfEmployed\SelfEmployedInterface](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md)

**Returns:** bool - Признак тестовой операции


<a name="method_setTest" class="anchor"></a>
#### public setTest() : self

```php
public setTest(bool $test) : self
```

**Summary**

Устанавливает признак тестовой операции.

**Details:**
* Inherited From: [\YooKassa\Model\SelfEmployed\SelfEmployedInterface](../classes/YooKassa-Model-SelfEmployed-SelfEmployedInterface.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">bool</code> | test  | Признак тестовой операции |

**Returns:** self - 




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