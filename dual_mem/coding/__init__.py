# -*- coding: utf-8 -*-
"""Coding/tool-use memory subsystem — separate write/store/search path for
engineering conversations that contain tool calls (Read/Edit/Bash/etc).

Flow: add() → has_tool_messages? → judge(is_coding?) → coding writer
      (extract → reconcile → coding store) OR chat path (normal extract).
"""
