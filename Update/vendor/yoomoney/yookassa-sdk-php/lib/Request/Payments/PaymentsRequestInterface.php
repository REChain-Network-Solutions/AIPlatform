<?php

/*
 * The MIT License
 *
 * Copyright (c) 2025 "YooMoney", NBСO LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

namespace YooKassa\Request\Payments;

use DateTime;

/**
 * Interface PaymentsRequestInterface.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property null|string $cursor Страница выдачи результатов, которую необходимо отобразить
 * @property null|DateTime $createdAtGte Время создания, от (включительно)
 * @property null|DateTime $createdAtGt Время создания, от (не включая)
 * @property null|DateTime $createdAtLte Время создания, до (включительно)
 * @property null|DateTime $createdAtLt Время создания, до (не включая)
 * @property null|DateTime $capturedAtGte Время подтверждения, от (включительно)
 * @property null|DateTime $capturedAtGt Время подтверждения, от (не включая)
 * @property null|DateTime $capturedAtLte Время подтверждения, до (включительно)
 * @property null|DateTime $capturedAtLt Время подтверждения, до (не включая)
 * @property null|int $limit Ограничение количества объектов платежа, отображаемых на одной странице выдачи
 * @property null|string $recipientGatewayId Идентификатор шлюза.
 * @property null|string $status Статус платежа
 */
interface PaymentsRequestInterface
{
    /**
     * Возвращает страницу выдачи результатов или null, если она до этого не была установлена.
     *
     * @return null|string Страница выдачи результатов
     */
    public function getCursor(): ?string;

    /**
     * Проверяет, была ли установлена страница выдачи результатов.
     *
     * @return bool True если страница выдачи результатов была установлена, false если нет
     */
    public function hasCursor(): bool;

    /**
     * Устанавливает страницу выдачи результатов.
     *
     * @param string $cursor Страница
     *
     * @return self
     */
    public function setCursor(string $cursor): self;

    /**
     * Возвращает ограничение количества объектов платежа или null, если оно до этого не было установлено.
     *
     * @return null|int Ограничение количества объектов платежа
     */
    public function getLimit(): ?int;

    /**
     * Проверяет, было ли установлено ограничение количества объектов платежа.
     *
     * @return bool True если ограничение количества объектов платежа было установлено, false если нет
     */
    public function hasLimit(): bool;

    /**
     * Устанавливает ограничение количества объектов платежа.
     *
     * @param int $limit Количества объектов платежа на странице
     *
     * @return self
     */
    public function setLimit(int $limit): self;

    /**
     * Возвращает дату создания от которой будут возвращены платежи или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, от (включительно)
     */
    public function getCreatedAtGte(): ?DateTime;

    /**
     * Проверяет, была ли установлена дата создания от которой выбираются платежи.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCreatedAtGte(): bool;

    /**
     * Устанавливает дату создания от которой выбираются платежи.
     *
     * @param DateTime|string|null $_created_at_gte Дата
     *
     * @return self
     */
    public function setCreatedAtGte(mixed $_created_at_gte): self;

    /**
     * Возвращает дату создания от которой будут возвращены платежи или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, от (не включая)
     */
    public function getCreatedAtGt(): ?DateTime;

    /**
     * Проверяет, была ли установлена дата создания от которой выбираются платежи.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCreatedAtGt(): bool;

    /**
     * Устанавливает дату создания от которой выбираются платежи.
     *
     * @param DateTime|string|null $created_at_gt Дата создания
     *
     * @return self
     */
    public function setCreatedAtGt(mixed $created_at_gt): self;

    /**
     * Возвращает дату создания до которой будут возвращены платежи или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, до (включительно)
     */
    public function getCreatedAtLte(): ?DateTime;

    /**
     * Проверяет, была ли установлена дата создания до которой выбираются платежи.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCreatedAtLte(): bool;

    /**
     * Устанавливает дату создания до которой выбираются платежи.
     *
     * @param DateTime|string|null $created_at_lte Дата
     *
     * @return self
     */
    public function setCreatedAtLte(mixed $created_at_lte): self;

    /**
     * Возвращает дату создания до которой будут возвращены платежи или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, до (не включая)
     */
    public function getCreatedAtLt(): ?DateTime;

    /**
     * Проверяет, была ли установлена дата создания до которой выбираются платежи.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCreatedAtLt(): bool;

    /**
     * Устанавливает дату создания до которой выбираются платежи.
     *
     * @param DateTime|string|null $created_at_lt Дата
     *
     * @return self
     */
    public function setCreatedAtLt(mixed $created_at_lt): self;

    /**
     * Возвращает дату создания от которой будут возвращены платежи или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, от (включительно)
     */
    public function getCapturedAtGte(): ?DateTime;

    /**
     * Проверяет, была ли установлена дата создания от которой выбираются платежи.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCapturedAtGte(): bool;

    /**
     * Устанавливает дату создания от которой выбираются платежи.
     *
     * @param DateTime|string|null $captured_at_gte Дата
     *
     * @return self
     */
    public function setCapturedAtGte(mixed $captured_at_gte): self;

    /**
     * Возвращает дату создания от которой будут возвращены платежи или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, от (не включая)
     */
    public function getCapturedAtGt(): ?DateTime;

    /**
     * Проверяет, была ли установлена дата создания от которой выбираются платежи.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCapturedAtGt(): bool;

    /**
     * Устанавливает дату создания от которой выбираются платежи.
     *
     * @param DateTime|string|null $captured_at_gt Дата
     *
     * @return self
     */
    public function setCapturedAtGt(mixed $captured_at_gt): self;

    /**
     * Возвращает дату создания до которой будут возвращены платежи или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, до (включительно)
     */
    public function getCapturedAtLte(): ?DateTime;

    /**
     * Проверяет, была ли установлена дата создания до которой выбираются платежи.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCapturedAtLte(): bool;

    /**
     * Устанавливает дату создания до которой выбираются платежи.
     *
     * @param DateTime|string|null $captured_at_lte Дата
     *
     * @return self
     */
    public function setCapturedAtLte(mixed $captured_at_lte): self;

    /**
     * Возвращает дату создания до которой будут возвращены платежи или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, до (не включая)
     */
    public function getCapturedAtLt(): ?DateTime;

    /**
     * Проверяет, была ли установлена дата создания до которой выбираются платежи.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCapturedAtLt(): bool;

    /**
     * Устанавливает дату создания до которой выбираются платежи.
     *
     * @param DateTime|string|null $captured_at_lt Дата
     *
     * @return self
     */
    public function setCapturedAtLt(mixed $captured_at_lt): self;

    /**
     * Возвращает статус выбираемых платежей или null, если он до этого не был установлен.
     *
     * @return null|string Статус выбираемых платежей
     */
    public function getStatus(): ?string;

    /**
     * Проверяет, был ли установлен статус выбираемых платежей.
     *
     * @return bool True если статус был установлен, false если нет
     */
    public function hasStatus(): bool;

    /**
     * Устанавливает статус выбираемых платежей.
     *
     * @param string $status Статус платежей
     *
     * @return self
     */
    public function setStatus(string $status): self;

    /**
     * Возвращает код способа оплаты выбираемых платежей или null, если он до этого не был установлен.
     *
     * @return null|string Код способа оплаты выбираемых платежей
     */
    public function getPaymentMethod(): ?string;

    /**
     * Проверяет, был ли установлен код способа оплаты выбираемых платежей.
     *
     * @return bool True если код способа оплаты был установлен, false если нет
     */
    public function hasPaymentMethod(): bool;

    /**
     * Устанавливает код способа оплаты выбираемых платежей.
     *
     * @param string $payment_method Код способа оплаты
     *
     * @return self
     */
    public function setPaymentMethod(string $payment_method): self;
}
