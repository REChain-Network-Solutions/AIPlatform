# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Helpers\Random
### Namespace: [\YooKassa\Helpers](../namespaces/yookassa-helpers.md)
---
**Summary:**

Класс, представляющий модель Random.

**Description:**

Класс хэлпера для генерации случайных значений, используется в тестах.

---
### Constants
* No constants found

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [bool()](../classes/YooKassa-Helpers-Random.md#method_bool) |  | Возвращает рандомное буллево значение. |
| public | [bytes()](../classes/YooKassa-Helpers-Random.md#method_bytes) |  | Возвращает рандомную последовательность байт |
| public | [float()](../classes/YooKassa-Helpers-Random.md#method_float) |  | Возвращает рандомное число с плавающей точкой. По умолчанию возвращает значение в промежутке от нуля до едениы. |
| public | [hex()](../classes/YooKassa-Helpers-Random.md#method_hex) |  | Возвращает строку, состоящую из символов '0123456789abcdef'. |
| public | [int()](../classes/YooKassa-Helpers-Random.md#method_int) |  | Возвращает рандомное целое число. По умолчанию возвращает число от нуля до PHP_INT_MAX. |
| public | [str()](../classes/YooKassa-Helpers-Random.md#method_str) |  | Возвращает строку из рандомных символов. |
| public | [value()](../classes/YooKassa-Helpers-Random.md#method_value) |  | Возвращает рандомное значение из переданного массива. |

---
### Details
* File: [lib/Helpers/Random.php](../../lib/Helpers/Random.php)
* Package: YooKassa\Helpers
* Class Hierarchy:
  * \YooKassa\Helpers\Random

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Methods
<a name="method_bool" class="anchor"></a>
#### public bool() : bool

```php
Static public bool() : bool
```

**Summary**

Возвращает рандомное буллево значение.

**Details:**
* Inherited From: [\YooKassa\Helpers\Random](../classes/YooKassa-Helpers-Random.md)

**Returns:** bool - Либо true либо false, одно из двух


<a name="method_bytes" class="anchor"></a>
#### public bytes() : string

```php
Static public bytes(int $length) : string
```

**Summary**

Возвращает рандомную последовательность байт

**Details:**
* Inherited From: [\YooKassa\Helpers\Random](../classes/YooKassa-Helpers-Random.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int</code> | length  | Длина возвращаемой строки |

**Returns:** string - Строка, состоящая из рандомных символов


<a name="method_float" class="anchor"></a>
#### public float() : float

```php
Static public float(null|float $min = null, null|float $max = null) : float
```

**Summary**

Возвращает рандомное число с плавающей точкой. По умолчанию возвращает значение в промежутке от нуля до едениы.

**Details:**
* Inherited From: [\YooKassa\Helpers\Random](../classes/YooKassa-Helpers-Random.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR float</code> | min  | Минимально возможное значение |
| <code lang="php">null OR float</code> | max  | Максимально возможное значение |

**Returns:** float - Рандомное число с плавающей точкой


<a name="method_hex" class="anchor"></a>
#### public hex() : string

```php
Static public hex(int $length, bool $useBest = true) : string
```

**Summary**

Возвращает строку, состоящую из символов '0123456789abcdef'.

**Details:**
* Inherited From: [\YooKassa\Helpers\Random](../classes/YooKassa-Helpers-Random.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int</code> | length  | Длина возвращаемой строки |
| <code lang="php">bool</code> | useBest  | Использовать ли функцию random_int если она доступна |

**Returns:** string - Строка, состоящая из рандомных символов


<a name="method_int" class="anchor"></a>
#### public int() : int

```php
Static public int(null|int $min = null, null|int $max = null) : int
```

**Summary**

Возвращает рандомное целое число. По умолчанию возвращает число от нуля до PHP_INT_MAX.

**Details:**
* Inherited From: [\YooKassa\Helpers\Random](../classes/YooKassa-Helpers-Random.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR int</code> | min  | Минимально возможное значение |
| <code lang="php">null OR int</code> | max  | Максимально возможное значение |

**Returns:** int - Рандомное целое число


<a name="method_str" class="anchor"></a>
#### public str() : string

```php
Static public str(int $length, null|int|string $maxLength = null, null|array|string $characters = null) : string
```

**Summary**

Возвращает строку из рандомных символов.

**Details:**
* Inherited From: [\YooKassa\Helpers\Random](../classes/YooKassa-Helpers-Random.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int</code> | length  | Длина возвращаемой строки, или минимальная длина, если передан парамтр $maxLength |
| <code lang="php">null OR int OR string</code> | maxLength  | Если параметр не равен null, возвращает сроку длиной от $length до $maxLength |
| <code lang="php">null OR array OR string</code> | characters  | Строка или массив используемых в строке символов |

**Returns:** string - Строка, состоящая из рандомных символов


<a name="method_value" class="anchor"></a>
#### public value() : mixed

```php
Static public value(array $values) : mixed
```

**Summary**

Возвращает рандомное значение из переданного массива.

**Details:**
* Inherited From: [\YooKassa\Helpers\Random](../classes/YooKassa-Helpers-Random.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array</code> | values  | Массив источник данных |

**Returns:** mixed - Случайное значение из переданного массива



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